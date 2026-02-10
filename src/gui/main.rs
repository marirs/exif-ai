#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::mpsc;

use eframe::egui;

use exif_ai::ai::local::download_model;
use exif_ai::config::Config;
use exif_ai::exif::{self, ExifData};
use exif_ai::pipeline::{collect_images, ImageKind, Pipeline, ProcessResult};

fn load_icon() -> Option<egui::IconData> {
    let png_bytes = include_bytes!("../../assets/icon_256.png");
    let img = image::load_from_memory(png_bytes).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::IconData {
        rgba: img.into_raw(),
        width: w,
        height: h,
    })
}

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1100.0, 720.0])
        .with_min_inner_size([800.0, 500.0])
        .with_drag_and_drop(true);

    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "exif-ai",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

// ── Messages sent from background threads to the UI ─────────────────

enum BgMessage {
    /// AI processing finished for one image.
    ProcessResult(ProcessResult),
    /// All images in the batch are done.
    BatchDone,
    /// Model download progress / completion.
    DownloadStatus(String),
    /// An error from a background task.
    Error(String),
}

// ── Per-image state shown in the UI ─────────────────────────────────

struct ImageEntry {
    path: PathBuf,
    kind: Option<ImageKind>,
    existing_exif: Option<ExifData>,
    result: Option<ProcessResult>,
    /// Texture handle for the preview thumbnail.
    texture: Option<egui::TextureHandle>,
}

// ── Tabs ────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Process,
    Settings,
}

// ── Main application state ──────────────────────────────────────────

struct App {
    config: Config,
    config_path: Option<PathBuf>,
    images: Vec<ImageEntry>,
    selected: Option<usize>,
    tab: Tab,
    processing: bool,
    status: String,
    dry_run: bool,
    rx: mpsc::Receiver<BgMessage>,
    tx: mpsc::Sender<BgMessage>,
    /// Tokio runtime for async tasks.
    rt: tokio::runtime::Runtime,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = mpsc::channel();
        let config = Config::load(None).unwrap_or_default();

        Self {
            config,
            config_path: None,
            images: Vec::new(),
            selected: None,
            tab: Tab::Process,
            processing: false,
            status: "Ready — drop images or click Open".into(),
            dry_run: false,
            rx,
            tx,
            rt: tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"),
        }
    }

    fn add_paths(&mut self, paths: Vec<PathBuf>) {
        let collected = collect_images(&paths);
        for path in collected {
            if self.images.iter().any(|e| e.path == path) {
                continue;
            }
            let kind = ImageKind::from_path(&path);
            let existing_exif = exif::read_exif(&path).ok();
            self.images.push(ImageEntry {
                path,
                kind,
                existing_exif,
                result: None,
                texture: None,
            });
        }
        if !self.images.is_empty() && self.selected.is_none() {
            self.selected = Some(0);
        }
        self.status = format!("{} image(s) loaded", self.images.len());
    }

    fn open_files(&mut self) {
        if let Some(paths) = rfd::FileDialog::new()
            .add_filter("Images", &[
                "jpg", "jpeg", "png", "webp", "tif", "tiff",
                "heic", "heif", "avif",
                "cr3", "cr2", "dng", "nef", "arw", "raf", "orf", "rw2", "pef", "srw",
            ])
            .pick_files()
        {
            self.add_paths(paths);
        }
    }

    fn open_folder(&mut self) {
        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
            self.add_paths(vec![dir]);
        }
    }

    fn start_processing(&mut self) {
        if self.images.is_empty() || self.processing {
            return;
        }
        self.processing = true;
        self.status = "Processing...".into();

        // Clear previous results
        for entry in &mut self.images {
            entry.result = None;
        }

        let paths: Vec<PathBuf> = self.images.iter().map(|e| e.path.clone()).collect();
        let config = self.config.clone();
        let dry_run = self.dry_run;
        let tx = self.tx.clone();

        self.rt.spawn(async move {
            let pipeline = Pipeline::builder()
                .from_config(&config)
                .dry_run(dry_run)
                .build();

            let pipeline = match pipeline {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.send(BgMessage::Error(format!("Pipeline build failed: {e}")));
                    let _ = tx.send(BgMessage::BatchDone);
                    return;
                }
            };

            for path in &paths {
                let result = pipeline.process_image(path).await;
                let _ = tx.send(BgMessage::ProcessResult(result));
            }
            let _ = tx.send(BgMessage::BatchDone);
        });
    }

    fn start_download(&mut self) {
        if self.processing {
            return;
        }
        self.processing = true;
        self.status = "Downloading BLIP model...".into();

        let model_path = self.config.ai_services.local.model_path.clone();
        let tx = self.tx.clone();

        self.rt.spawn(async move {
            let dir = if model_path.is_empty() {
                None
            } else {
                Some(std::path::Path::new(&model_path).to_path_buf())
            };
            match download_model(dir.as_deref()).await {
                Ok(p) => {
                    let _ = tx.send(BgMessage::DownloadStatus(format!(
                        "Model downloaded to: {}",
                        p.display()
                    )));
                }
                Err(e) => {
                    let _ = tx.send(BgMessage::Error(format!("Download failed: {e}")));
                }
            }
            let _ = tx.send(BgMessage::BatchDone);
        });
    }

    fn poll_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                BgMessage::ProcessResult(result) => {
                    if let Some(entry) = self
                        .images
                        .iter_mut()
                        .find(|e| e.path == result.path)
                    {
                        entry.existing_exif = Some(result.existing_exif.clone());
                        entry.result = Some(result);
                    }
                }
                BgMessage::BatchDone => {
                    self.processing = false;
                    let ok = self.images.iter().filter(|e| {
                        e.result.as_ref().is_some_and(|r| r.error.is_none())
                    }).count();
                    let fail = self.images.iter().filter(|e| {
                        e.result.as_ref().is_some_and(|r| r.error.is_some())
                    }).count();
                    self.status = format!("Done — {ok} succeeded, {fail} failed");
                }
                BgMessage::DownloadStatus(msg) => {
                    self.processing = false;
                    self.status = msg;
                }
                BgMessage::Error(msg) => {
                    self.processing = false;
                    self.status = format!("Error: {msg}");
                }
            }
        }
    }

}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_messages();

        // Request repaint while processing so we pick up messages
        if self.processing {
            ctx.request_repaint();
        }

        // Handle dropped files
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            self.add_paths(dropped);
        }

        // ── Top bar ─────────────────────────────────────────────────
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("exif-ai");
                ui.separator();

                let process_tab = ui.selectable_label(self.tab == Tab::Process, "📷 Process");
                let settings_tab = ui.selectable_label(self.tab == Tab::Settings, "⚙ Settings");
                if process_tab.clicked() {
                    self.tab = Tab::Process;
                }
                if settings_tab.clicked() {
                    self.tab = Tab::Settings;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.processing {
                        ui.spinner();
                    }
                    ui.label(&self.status);
                });
            });
        });

        match self.tab {
            Tab::Process => self.show_process_tab(ctx),
            Tab::Settings => self.show_settings_tab(ctx),
        }
    }
}

// ── Process tab ─────────────────────────────────────────────────────

impl App {
    fn show_process_tab(&mut self, ctx: &egui::Context) {
        // ── Bottom toolbar ──────────────────────────────────────────
        egui::TopBottomPanel::bottom("toolbar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.processing, egui::Button::new("📂 Open Files")).clicked() {
                    self.open_files();
                }
                if ui.add_enabled(!self.processing, egui::Button::new("📁 Open Folder")).clicked() {
                    self.open_folder();
                }
                ui.separator();

                if ui.add_enabled(
                    !self.processing && !self.images.is_empty(),
                    egui::Button::new("▶ Process"),
                ).clicked() {
                    self.dry_run = false;
                    self.start_processing();
                }
                if ui.add_enabled(
                    !self.processing && !self.images.is_empty(),
                    egui::Button::new("👁 Dry Run"),
                ).clicked() {
                    self.dry_run = true;
                    self.start_processing();
                }

                ui.separator();
                if ui.add_enabled(!self.processing, egui::Button::new("⬇ Download Model")).clicked() {
                    self.start_download();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add_enabled(!self.images.is_empty(), egui::Button::new("🗑 Clear All")).clicked() {
                        self.images.clear();
                        self.selected = None;
                        self.status = "Ready — drop images or click Open".into();
                    }
                });
            });
            ui.add_space(4.0);
        });

        // ── Left panel: image list ──────────────────────────────────
        egui::SidePanel::left("image_list")
            .default_width(260.0)
            .min_width(180.0)
            .show(ctx, |ui| {
                ui.heading("Images");
                ui.separator();

                if self.images.is_empty() {
                    ui.centered_and_justified(|ui| {
                        ui.label(egui::RichText::new("Drop images here\nor click Open")
                            .size(16.0)
                            .color(egui::Color32::GRAY));
                    });
                    return;
                }

                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut new_selected = self.selected;
                    for (i, entry) in self.images.iter().enumerate() {
                        let is_selected = self.selected == Some(i);
                        let filename = entry.path.file_name()
                            .map(|f| f.to_string_lossy().to_string())
                            .unwrap_or_else(|| entry.path.display().to_string());

                        let status_icon = match &entry.result {
                            Some(r) if r.error.is_none() => "✅ ",
                            Some(_) => "❌ ",
                            None => "  ",
                        };

                        let label = format!("{status_icon}{filename}");
                        let resp = ui.selectable_label(is_selected, &label);
                        if resp.clicked() {
                            new_selected = Some(i);
                        }
                    }
                    self.selected = new_selected;
                });
            });

        // ── Central panel: preview + results ────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(idx) = self.selected {
                if idx < self.images.len() {
                    // Load texture if needed
                    Self::load_texture_static(ctx, &mut self.images[idx]);

                    let entry = &self.images[idx];
                    let has_result = entry.result.is_some();
                    let processing = self.processing;

                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                        // Image preview
                        ui.horizontal(|ui| {
                            if let Some(ref tex) = entry.texture {
                                let size = tex.size_vec2();
                                let max_h = 300.0;
                                let scale = (max_h / size.y).min(1.0);
                                ui.image(egui::load::SizedTexture::new(
                                    tex.id(),
                                    size * scale,
                                ));
                            }

                            ui.vertical(|ui| {
                                ui.heading(entry.path.file_name()
                                    .map(|f| f.to_string_lossy().to_string())
                                    .unwrap_or_default());
                                ui.label(format!("Path: {}", entry.path.display()));
                                if let Some(kind) = entry.kind {
                                    ui.label(format!("Format: {kind:?}"));
                                }
                                ui.add_space(8.0);

                                // Existing EXIF summary
                                if let Some(ref exif_data) = entry.existing_exif {
                                    Self::show_existing_exif(ui, exif_data);
                                }
                            });
                        });

                        ui.add_space(12.0);
                        ui.separator();

                        // AI results
                        if has_result {
                            let result = entry.result.as_ref().unwrap();
                            Self::show_ai_results(ui, result, self.dry_run);
                        } else if processing {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("Processing...");
                            });
                        }
                    });
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Select an image from the list")
                        .size(18.0)
                        .color(egui::Color32::GRAY));
                });
            }
        });
    }

    fn load_texture_static(ctx: &egui::Context, entry: &mut ImageEntry) {
        if entry.texture.is_some() {
            return;
        }

        // Try loading directly with the image crate (JPEG, PNG, WebP, TIFF)
        let decoded = std::fs::read(&entry.path).ok().and_then(|bytes| {
            image::load_from_memory(&bytes).ok()
        }).or_else(|| {
            // Fallback: use macOS `sips` to convert HEIC/RAW/AVIF to JPEG for preview
            #[cfg(target_os = "macos")]
            {
                let tmp = std::env::temp_dir().join("exif_ai_preview.jpg");
                let status = std::process::Command::new("sips")
                    .args(["-s", "format", "jpeg", "-s", "formatOptions", "70"])
                    .arg(&entry.path)
                    .arg("--out")
                    .arg(&tmp)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
                if status.is_ok_and(|s| s.success()) {
                    if let Ok(bytes) = std::fs::read(&tmp) {
                        let _ = std::fs::remove_file(&tmp);
                        return image::load_from_memory(&bytes).ok();
                    }
                }
            }
            None
        });

        if let Some(img) = decoded {
            let img = img.thumbnail(400, 400);
            let size = [img.width() as usize, img.height() as usize];
            let rgba = img.to_rgba8();
            let pixels = rgba.as_flat_samples();
            let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
            entry.texture = Some(ctx.load_texture(
                entry.path.to_string_lossy(),
                color_image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn show_existing_exif(ui: &mut egui::Ui, data: &ExifData) {
        egui::CollapsingHeader::new(egui::RichText::new("Existing EXIF").strong())
            .default_open(true)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(220.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let section = |ui: &mut egui::Ui, heading: &str, fields: &[(&str, Option<&str>)]| {
                            let any = fields.iter().any(|(_, v)| v.is_some());
                            if !any { return; }
                            ui.label(egui::RichText::new(heading).small().color(egui::Color32::GRAY));
                            ui.end_row();
                            for &(label, ref value) in fields {
                                if let Some(val) = value {
                                    ui.label(egui::RichText::new(label).strong());
                                    ui.label(*val);
                                    ui.end_row();
                                }
                            }
                        };

                        egui::Grid::new("existing_exif_grid")
                            .num_columns(2)
                            .spacing([12.0, 4.0])
                            .show(ui, |ui| {
                                // Camera
                                section(ui, "CAMERA", &[
                                    ("Make", data.make.as_deref()),
                                    ("Model", data.model.as_deref()),
                                    ("Lens", data.lens_model.as_deref()),
                                ]);

                                // Exposure
                                section(ui, "EXPOSURE", &[
                                    ("Date", data.date_time.as_deref()),
                                    ("Exposure", data.exposure_time.as_deref()),
                                    ("F-Number", data.f_number.as_deref()),
                                    ("ISO", data.iso.as_deref()),
                                    ("Focal Length", data.focal_length.as_deref()),
                                ]);

                                // Image
                                section(ui, "IMAGE", &[
                                    ("Width", data.image_width.as_deref()),
                                    ("Height", data.image_height.as_deref()),
                                    ("Orientation", data.orientation.as_deref()),
                                    ("Color Space", data.color_space.as_deref()),
                                    ("X Resolution", data.x_resolution.as_deref()),
                                    ("Y Resolution", data.y_resolution.as_deref()),
                                    ("Software", data.software.as_deref()),
                                ]);

                                // Metadata (IPTC / XMP fields)
                                section(ui, "METADATA", &[
                                    ("Title", data.title.as_deref()),
                                    ("Description", data.description.as_deref()),
                                    ("Keywords", data.keywords.as_deref()),
                                    ("Subject", data.subject.as_deref()),
                                ]);

                                // GPS
                                if data.has_gps {
                                    if let (Some(lat), Some(lon)) = (data.gps_latitude, data.gps_longitude) {
                                        ui.label(egui::RichText::new("GPS").small().color(egui::Color32::GRAY));
                                        ui.end_row();
                                        ui.label(egui::RichText::new("Coordinates").strong());
                                        ui.label(format!("{lat:.6}, {lon:.6}"));
                                        ui.end_row();
                                    }
                                }
                            });
                    });
            });
    }

    fn show_ai_results(ui: &mut egui::Ui, result: &ProcessResult, dry_run: bool) {
        if let Some(ref err) = result.error {
            ui.colored_label(egui::Color32::from_rgb(220, 50, 50), format!("Error: {err}"));
            return;
        }

        // Status badge helper
        let status_label = |ui: &mut egui::Ui, written: bool| {
            if written {
                if dry_run {
                    ui.label(
                        egui::RichText::new("⟡ would write")
                            .color(egui::Color32::from_rgb(100, 160, 255))
                            .italics(),
                    );
                } else {
                    ui.colored_label(egui::Color32::from_rgb(50, 180, 50), "✓ written");
                }
            }
        };

        if let Some(ref service) = result.ai_service_used {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("AI Service:").strong().size(15.0));
                ui.label(
                    egui::RichText::new(service)
                        .strong()
                        .size(15.0)
                        .color(egui::Color32::from_rgb(100, 180, 255)),
                );
            });
        }
        ui.add_space(8.0);

        if let Some(ref ai) = result.ai_result {
            egui::Grid::new("ai_results_grid")
                .num_columns(2)
                .spacing([12.0, 12.0])
                .show(ui, |ui| {
                    if let Some(ref title) = ai.title {
                        ui.label(egui::RichText::new("Title").strong());
                        ui.horizontal(|ui| {
                            ui.label(title);
                            status_label(ui, result.title_written);
                        });
                        ui.end_row();
                    }

                    if let Some(ref desc) = ai.description {
                        ui.label(egui::RichText::new("Description").strong());
                        ui.horizontal_wrapped(|ui| {
                            ui.label(desc);
                            status_label(ui, result.description_written);
                        });
                        ui.end_row();
                    }

                    if let Some(ref tags) = ai.tags {
                        ui.label(egui::RichText::new("Tags").strong());
                        ui.horizontal_wrapped(|ui| {
                            for tag in tags {
                                ui.label(
                                    egui::RichText::new(tag)
                                        .background_color(egui::Color32::from_rgb(60, 60, 80))
                                        .color(egui::Color32::WHITE),
                                );
                            }
                            status_label(ui, result.tags_written);
                        });
                        ui.end_row();
                    }

                    if let Some(ref gps) = ai.gps {
                        ui.label(egui::RichText::new("GPS").strong());
                        ui.horizontal(|ui| {
                            ui.label(format!("{:.6}, {:.6}", gps.latitude, gps.longitude));
                            status_label(ui, result.gps_written);
                        });
                        ui.end_row();
                    }

                    if let Some(ref subjects) = ai.subject {
                        if !subjects.is_empty() {
                            ui.label(egui::RichText::new("Subject").strong());
                            ui.horizontal_wrapped(|ui| {
                                ui.label(subjects.join(", "));
                                status_label(ui, result.subject_written);
                            });
                            ui.end_row();
                        }
                    }
                });

            // Skipped fields
            if !result.skipped_fields.is_empty() {
                ui.add_space(8.0);
                ui.colored_label(
                    egui::Color32::from_rgb(180, 180, 50),
                    format!("Skipped: {}", result.skipped_fields.join(", ")),
                );
            }

            // Sidecar
            if let Some(ref sidecar) = result.sidecar_path {
                ui.add_space(4.0);
                ui.label(format!("Sidecar XMP: {}", sidecar.display()));
            }
        }
    }
}

// ── Settings tab ────────────────────────────────────────────────────

impl App {
    fn show_settings_tab(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Configuration");
                ui.add_space(8.0);

                // Config file path
                ui.horizontal(|ui| {
                    ui.label("Config file:");
                    if let Some(ref path) = self.config_path {
                        ui.label(path.display().to_string());
                    } else {
                        ui.label("(default)");
                    }
                    if ui.button("Load...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("JSON", &["json"])
                            .pick_file()
                        {
                            match Config::load(Some(&path)) {
                                Ok(c) => {
                                    self.config = c;
                                    self.config_path = Some(path);
                                    self.status = "Config loaded".into();
                                }
                                Err(e) => {
                                    self.status = format!("Failed to load config: {e}");
                                }
                            }
                        }
                    }
                    if ui.button("Save").clicked() {
                        let path = self.config_path.as_deref();
                        match self.config.save(path) {
                            Ok(()) => self.status = "Config saved".into(),
                            Err(e) => self.status = format!("Failed to save config: {e}"),
                        }
                    }
                });

                ui.add_space(16.0);
                ui.separator();

                // ── AI Services ─────────────────────────────────────
                ui.add_space(8.0);
                ui.heading("AI Services");
                ui.add_space(4.0);

                // Service order
                ui.horizontal(|ui| {
                    ui.label("Service order:");
                    ui.label(
                        egui::RichText::new(self.config.service_order.join(" → "))
                            .monospace(),
                    );
                });
                ui.add_space(8.0);

                // Local
                egui::CollapsingHeader::new(egui::RichText::new("Local (BLIP)").strong())
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.checkbox(&mut self.config.ai_services.local.enabled, "Enabled");
                        ui.horizontal(|ui| {
                            ui.label("Model path:");
                            ui.text_edit_singleline(&mut self.config.ai_services.local.model_path);
                        });
                    });

                // OpenAI
                egui::CollapsingHeader::new(egui::RichText::new("OpenAI").strong())
                    .default_open(self.config.ai_services.openai.enabled)
                    .show(ui, |ui| {
                        ui.checkbox(&mut self.config.ai_services.openai.enabled, "Enabled");
                        ui.horizontal(|ui| {
                            ui.label("API Key:");
                            ui.add(egui::TextEdit::singleline(&mut self.config.ai_services.openai.api_key).password(true));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Model:");
                            ui.text_edit_singleline(&mut self.config.ai_services.openai.model);
                        });
                    });

                // Gemini
                egui::CollapsingHeader::new(egui::RichText::new("Gemini").strong())
                    .default_open(self.config.ai_services.gemini.enabled)
                    .show(ui, |ui| {
                        ui.checkbox(&mut self.config.ai_services.gemini.enabled, "Enabled");
                        ui.horizontal(|ui| {
                            ui.label("API Key:");
                            ui.add(egui::TextEdit::singleline(&mut self.config.ai_services.gemini.api_key).password(true));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Model:");
                            ui.text_edit_singleline(&mut self.config.ai_services.gemini.model);
                        });
                    });

                // Cloudflare
                egui::CollapsingHeader::new(egui::RichText::new("Cloudflare").strong())
                    .default_open(self.config.ai_services.cloudflare.enabled)
                    .show(ui, |ui| {
                        ui.checkbox(&mut self.config.ai_services.cloudflare.enabled, "Enabled");
                        ui.horizontal(|ui| {
                            ui.label("Account ID:");
                            ui.text_edit_singleline(&mut self.config.ai_services.cloudflare.account_id);
                        });
                        ui.horizontal(|ui| {
                            ui.label("API Token:");
                            ui.add(egui::TextEdit::singleline(&mut self.config.ai_services.cloudflare.api_token).password(true));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Model:");
                            ui.text_edit_singleline(&mut self.config.ai_services.cloudflare.model);
                        });
                    });

                ui.add_space(16.0);
                ui.separator();

                // ── EXIF Fields ─────────────────────────────────────
                ui.add_space(8.0);
                ui.heading("EXIF Fields");
                ui.add_space(4.0);

                ui.checkbox(&mut self.config.exif_fields.write_title, "Write title (ImageDescription + XPTitle)");
                ui.checkbox(&mut self.config.exif_fields.write_description, "Write description (UserComment + XPComment)");
                ui.checkbox(&mut self.config.exif_fields.write_tags, "Write tags (XPKeywords)");
                ui.checkbox(&mut self.config.exif_fields.write_gps, "Write GPS coordinates");
                ui.checkbox(&mut self.config.exif_fields.write_subject, "Write subject (XPSubject)");
                ui.add_space(4.0);
                ui.checkbox(&mut self.config.exif_fields.overwrite_existing, "Overwrite existing values");

                ui.add_space(16.0);
                ui.separator();

                // ── Output ──────────────────────────────────────────
                ui.add_space(8.0);
                ui.heading("Output");
                ui.add_space(4.0);

                ui.checkbox(&mut self.config.output.dry_run, "Dry run (preview only)");
                ui.checkbox(&mut self.config.output.backup_originals, "Backup originals (.bak)");
            });
        });
    }
}
