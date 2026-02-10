use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::ai::{self, AiResult, AiService};
use crate::config::Config;
use crate::exif::{self, ExifData};
use crate::exif::write_exif;

/// Supported image extensions.
const IMAGE_EXTENSIONS: &[&str] = &[
    // Native write support (EXIF+XMP+IPTC)
    "jpg", "jpeg",
    // Native write support (XMP)
    "png", "webp",
    // Native write support (EXIF)
    "tif", "tiff",
    // HEIC/HEIF — read EXIF, sidecar XMP write
    "heic", "heif",
    // AVIF — read EXIF, sidecar XMP write
    "avif",
    // RAW formats — read EXIF, sidecar XMP write
    "cr3", "cr2", "dng", "nef", "arw", "raf", "orf", "rw2", "pef", "srw",
];

/// The write strategy for a given image file, determined by its format.
///
/// Different image formats support different metadata embedding approaches:
/// - **Native** formats (JPEG, PNG, WebP, TIFF) have metadata written directly into the file.
/// - **Sidecar** formats (HEIC, AVIF, RAW) get a `.xmp` sidecar file written alongside the original.
///
/// Use [`ImageKind::from_path`] to detect the format from a file extension.
///
/// # Example
///
/// ```rust
/// use exif_ai::pipeline::ImageKind;
/// use std::path::Path;
///
/// let kind = ImageKind::from_path(Path::new("photo.heic"));
/// assert_eq!(kind, Some(ImageKind::Sidecar));
///
/// let kind = ImageKind::from_path(Path::new("photo.jpg"));
/// assert_eq!(kind, Some(ImageKind::Jpeg));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageKind {
    /// JPEG — full EXIF+XMP+IPTC write support
    Jpeg,
    /// PNG — XMP in iTXt chunk
    Png,
    /// WebP — EXIF+XMP in RIFF chunks
    WebP,
    /// TIFF — EXIF write via little_exif
    Tiff,
    /// HEIC/HEIF/AVIF/RAW — read EXIF from original, write sidecar .xmp
    Sidecar,
}

impl ImageKind {
    /// Determine the image kind from a file path extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "png" => Some(Self::Png),
            "webp" => Some(Self::WebP),
            "tif" | "tiff" => Some(Self::Tiff),
            "heic" | "heif" | "avif"
            | "cr3" | "cr2" | "dng" | "nef" | "arw" | "raf" | "orf" | "rw2" | "pef" | "srw"
                => Some(Self::Sidecar),
            _ => None,
        }
    }

    /// Get the MIME type for sending to AI services.
    pub fn mime_type(&self, path: &Path) -> &'static str {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "webp" => "image/webp",
            "tif" | "tiff" => "image/tiff",
            "heic" => "image/heic",
            "heif" => "image/heif",
            "avif" => "image/avif",
            "cr2" => "image/x-canon-cr2",
            "cr3" => "image/x-canon-cr3",
            "dng" => "image/x-adobe-dng",
            "nef" => "image/x-nikon-nef",
            "arw" => "image/x-sony-arw",
            "raf" => "image/x-fuji-raf",
            "orf" => "image/x-olympus-orf",
            "rw2" => "image/x-panasonic-rw2",
            "pef" => "image/x-pentax-pef",
            "srw" => "image/x-samsung-srw",
            _ => "image/jpeg",
        }
    }
}

/// The result of processing a single image through the AI pipeline.
///
/// Contains the AI analysis output, which metadata fields were written,
/// any errors encountered, and the sidecar path for HEIC/RAW formats.
///
/// # Example
///
/// ```rust,no_run
/// # use exif_ai::pipeline::{process_image, build_service_chain, ProcessResult};
/// # use exif_ai::config::Config;
/// # async fn example() {
/// # let config = Config::default();
/// # let services = build_service_chain(&config);
/// let result = process_image("photo.jpg".as_ref(), &services, &config).await;
///
/// if result.error.is_none() {
///     println!("Title written: {}", result.title_written);
///     println!("AI service: {:?}", result.ai_service_used);
///     if let Some(ref sidecar) = result.sidecar_path {
///         println!("Sidecar: {}", sidecar.display());
///     }
/// }
/// # }
/// ```
#[derive(Debug)]
pub struct ProcessResult {
    pub path: PathBuf,
    pub ai_result: Option<AiResult>,
    pub existing_exif: ExifData,
    pub title_written: bool,
    pub description_written: bool,
    pub tags_written: bool,
    pub gps_written: bool,
    pub subject_written: bool,
    pub skipped_fields: Vec<String>,
    pub error: Option<String>,
    pub ai_service_used: Option<String>,
    /// If a sidecar XMP file was written (for HEIC/RAW), this is the path.
    pub sidecar_path: Option<PathBuf>,
    /// The image kind detected for this file.
    pub image_kind: Option<ImageKind>,
}

/// Collect supported image files from the given paths.
///
/// Accepts a mix of file paths and directory paths. Directories are walked
/// recursively (following symlinks). Only files with supported image extensions
/// are included (see [`ImageKind`] for the full list).
///
/// # Example
///
/// ```rust,no_run
/// use exif_ai::pipeline::collect_images;
/// use std::path::PathBuf;
///
/// let images = collect_images(&[
///     PathBuf::from("photo.jpg"),       // single file
///     PathBuf::from("./photos/"),        // entire directory
/// ]);
/// println!("Found {} images", images.len());
/// ```
pub fn collect_images(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut images = Vec::new();

    for path in paths {
        if path.is_file() {
            if is_supported_image(path) {
                images.push(path.clone());
            } else {
                log::warn!("Skipping unsupported file: {}", path.display());
            }
        } else if path.is_dir() {
            for entry in WalkDir::new(path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let p = entry.path();
                if p.is_file() && is_supported_image(p) {
                    images.push(p.to_path_buf());
                }
            }
        } else {
            log::warn!("Path does not exist: {}", path.display());
        }
    }

    images
}

/// Check if a file has a supported image extension.
fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Create a backup of the original file.
fn backup_file(path: &Path) -> Result<PathBuf> {
    let backup_path = path.with_extension(format!(
        "{}.bak",
        path.extension().unwrap_or_default().to_string_lossy()
    ));

    if !backup_path.exists() {
        std::fs::copy(path, &backup_path).context("Failed to create backup")?;
        log::debug!("Backup created: {}", backup_path.display());
    }

    Ok(backup_path)
}

/// Build the AI service failover chain from configuration.
///
/// Creates a list of AI service instances based on the `service_order` and
/// enabled flags in the config. Services are tried in order during
/// [`process_image`] — if one fails, the next is attempted.
///
/// # Example
///
/// ```rust,no_run
/// use exif_ai::config::Config;
/// use exif_ai::pipeline::build_service_chain;
///
/// let config = Config::load(Some("config.json".as_ref())).unwrap();
/// let services = build_service_chain(&config);
/// println!("Configured {} AI services", services.len());
/// ```
pub fn build_service_chain(config: &Config) -> Vec<Box<dyn AiService>> {
    let mut services: Vec<Box<dyn AiService>> = Vec::new();

    for name in &config.service_order {
        match name.as_str() {
            "openai" if config.ai_services.openai.enabled => {
                if config.ai_services.openai.api_key.is_empty() {
                    log::warn!("OpenAI enabled but no API key configured");
                    continue;
                }
                services.push(Box::new(ai::OpenAiService::new(
                    config.ai_services.openai.api_key.clone(),
                    config.ai_services.openai.model.clone(),
                )));
            }
            "gemini" if config.ai_services.gemini.enabled => {
                if config.ai_services.gemini.api_key.is_empty() {
                    log::warn!("Gemini enabled but no API key configured");
                    continue;
                }
                services.push(Box::new(ai::GeminiService::new(
                    config.ai_services.gemini.api_key.clone(),
                    config.ai_services.gemini.model.clone(),
                )));
            }
            "cloudflare" if config.ai_services.cloudflare.enabled => {
                if config.ai_services.cloudflare.account_id.is_empty()
                    || config.ai_services.cloudflare.api_token.is_empty()
                {
                    log::warn!("Cloudflare enabled but account ID or API token not configured");
                    continue;
                }
                services.push(Box::new(ai::CloudflareService::new(
                    config.ai_services.cloudflare.account_id.clone(),
                    config.ai_services.cloudflare.api_token.clone(),
                    config.ai_services.cloudflare.model.clone(),
                )));
            }
            "local" if config.ai_services.local.enabled => {
                let local_service = if config.ai_services.local.model_path.is_empty() {
                    match ai::LocalService::from_default_dir() {
                        Ok(s) => s,
                        Err(e) => {
                            log::warn!("Local service: failed to resolve model directory: {e}");
                            continue;
                        }
                    }
                } else {
                    let dir = std::path::PathBuf::from(&config.ai_services.local.model_path);
                    ai::LocalService::new(
                        dir.join("model.safetensors"),
                        dir.join("tokenizer.json"),
                    )
                };
                if !local_service.model_exists() {
                    log::warn!(
                        "Local model not found. Run `exif-ai-cli --download-model` to download it. Skipping local service."
                    );
                    continue;
                }
                services.push(Box::new(local_service));
            }
            _ => {}
        }
    }

    services
}

/// Process a single image through the full AI pipeline.
///
/// This is the main entry point for the library. It performs the complete flow:
///
/// 1. **Read** — Extracts existing EXIF metadata from the image
/// 2. **Analyze** — Sends the image to AI services (with failover) for analysis
/// 3. **Write** — Writes AI-generated metadata back to the file (format-aware)
///
/// For JPEG, PNG, WebP, and TIFF files, metadata is written directly into the file.
/// For HEIC, AVIF, and RAW formats, a sidecar `.xmp` file is created alongside
/// the original (the original file is never modified).
///
/// # Arguments
///
/// * `path` — Path to the image file
/// * `services` — AI service chain from [`build_service_chain`]
/// * `config` — Configuration (controls which fields to write, dry run, backups, etc.)
///
/// # Returns
///
/// A [`ProcessResult`] containing the AI output, which fields were written,
/// any errors, and the sidecar path (if applicable).
///
/// # Example
///
/// ```rust,no_run
/// use exif_ai::config::Config;
/// use exif_ai::pipeline::{build_service_chain, process_image};
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = Config::load(Some("config.json".as_ref()))?;
/// let services = build_service_chain(&config);
///
/// let result = process_image(Path::new("photo.heic"), &services, &config).await;
/// if let Some(ref ai) = result.ai_result {
///     println!("Title: {:?}", ai.title);
///     println!("Tags: {:?}", ai.tags);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn process_image(
    path: &Path,
    services: &[Box<dyn AiService>],
    config: &Config,
) -> ProcessResult {
    let kind = ImageKind::from_path(path);

    let mut result = ProcessResult {
        path: path.to_path_buf(),
        ai_result: None,
        existing_exif: ExifData::default(),
        title_written: false,
        description_written: false,
        tags_written: false,
        gps_written: false,
        subject_written: false,
        skipped_fields: Vec::new(),
        error: None,
        ai_service_used: None,
        sidecar_path: None,
        image_kind: kind,
    };

    // Read existing EXIF
    match exif::read_exif(path) {
        Ok(data) => result.existing_exif = data,
        Err(e) => {
            log::warn!("Failed to read EXIF from {}: {e}", path.display());
        }
    }

    // Read and encode image
    let image_bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) => {
            result.error = Some(format!("Failed to read file: {e}"));
            return result;
        }
    };
    let image_base64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_bytes);

    // Determine MIME type for AI service
    let mime_type = kind
        .map(|k| k.mime_type(path))
        .unwrap_or("image/jpeg");

    // Build prompt
    let prompt = ai::build_prompt();

    // Try each AI service in order (failover chain)
    let mut errors = Vec::new();
    for service in services {
        log::info!("  Trying {}...", service.name());

        // Use file-based analysis for services that support it (e.g. local BLIP),
        // otherwise fall back to base64 analysis.
        let ai_response = if service.supports_file_analysis() {
            service.analyze_file(path)
        } else {
            service.analyze(&image_base64, &prompt, mime_type).await
        };

        match ai_response {
            Ok(ai_data) => {
                if ai_data.title.is_some() || ai_data.description.is_some() {
                    result.ai_result = Some(ai_data);
                    result.ai_service_used = Some(service.name().to_string());
                    log::info!("  {} succeeded", service.name());
                    break;
                } else {
                    errors.push(format!("{}: returned empty result", service.name()));
                }
            }
            Err(e) => {
                errors.push(format!("{}: {e}", service.name()));
                log::warn!("  {} failed: {e}", service.name());
            }
        }
    }

    if result.ai_result.is_none() {
        result.error = Some(format!(
            "All AI services failed: {}",
            errors.join("; ")
        ));
        return result;
    }

    // Backup original if configured
    if config.output.backup_originals && !config.output.dry_run {
        if let Err(e) = backup_file(path) {
            log::warn!("Failed to backup {}: {e}", path.display());
        }
    }

    // Write metadata based on image kind
    let ai_data = result.ai_result.as_ref().unwrap();
    let image_kind = kind.unwrap_or(ImageKind::Jpeg);

    match write_exif(
        path,
        ai_data,
        &result.existing_exif,
        &config.exif_fields,
        config.output.dry_run,
        image_kind,
    ) {
        Ok(write_result) => {
            result.title_written = write_result.title_written;
            result.description_written = write_result.description_written;
            result.tags_written = write_result.tags_written;
            result.gps_written = write_result.gps_written;
            result.subject_written = write_result.subject_written;
            result.skipped_fields = write_result.skipped_fields;
            result.sidecar_path = write_result.sidecar_path;
        }
        Err(e) => {
            result.error = Some(format!("Failed to write metadata: {e}"));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // ── ImageKind::from_path ──────────────────────────────────────────

    #[test]
    fn image_kind_jpeg() {
        assert_eq!(ImageKind::from_path(Path::new("photo.jpg")), Some(ImageKind::Jpeg));
        assert_eq!(ImageKind::from_path(Path::new("photo.jpeg")), Some(ImageKind::Jpeg));
        assert_eq!(ImageKind::from_path(Path::new("PHOTO.JPG")), Some(ImageKind::Jpeg));
    }

    #[test]
    fn image_kind_png() {
        assert_eq!(ImageKind::from_path(Path::new("image.png")), Some(ImageKind::Png));
        assert_eq!(ImageKind::from_path(Path::new("IMAGE.PNG")), Some(ImageKind::Png));
    }

    #[test]
    fn image_kind_webp() {
        assert_eq!(ImageKind::from_path(Path::new("image.webp")), Some(ImageKind::WebP));
    }

    #[test]
    fn image_kind_tiff() {
        assert_eq!(ImageKind::from_path(Path::new("scan.tif")), Some(ImageKind::Tiff));
        assert_eq!(ImageKind::from_path(Path::new("scan.tiff")), Some(ImageKind::Tiff));
    }

    #[test]
    fn image_kind_sidecar_heic() {
        assert_eq!(ImageKind::from_path(Path::new("photo.heic")), Some(ImageKind::Sidecar));
        assert_eq!(ImageKind::from_path(Path::new("photo.heif")), Some(ImageKind::Sidecar));
        assert_eq!(ImageKind::from_path(Path::new("photo.HEIC")), Some(ImageKind::Sidecar));
    }

    #[test]
    fn image_kind_sidecar_avif() {
        assert_eq!(ImageKind::from_path(Path::new("photo.avif")), Some(ImageKind::Sidecar));
    }

    #[test]
    fn image_kind_sidecar_raw() {
        for ext in &["cr3", "cr2", "dng", "nef", "arw", "raf", "orf", "rw2", "pef", "srw"] {
            let path = format!("photo.{ext}");
            assert_eq!(
                ImageKind::from_path(Path::new(&path)),
                Some(ImageKind::Sidecar),
                "Expected Sidecar for .{ext}"
            );
        }
    }

    #[test]
    fn image_kind_unsupported() {
        assert_eq!(ImageKind::from_path(Path::new("doc.pdf")), None);
        assert_eq!(ImageKind::from_path(Path::new("video.mp4")), None);
        assert_eq!(ImageKind::from_path(Path::new("noext")), None);
    }

    // ── ImageKind::mime_type ──────────────────────────────────────────

    #[test]
    fn mime_type_jpeg() {
        let kind = ImageKind::Jpeg;
        assert_eq!(kind.mime_type(Path::new("a.jpg")), "image/jpeg");
        assert_eq!(kind.mime_type(Path::new("a.jpeg")), "image/jpeg");
    }

    #[test]
    fn mime_type_png() {
        assert_eq!(ImageKind::Png.mime_type(Path::new("a.png")), "image/png");
    }

    #[test]
    fn mime_type_webp() {
        assert_eq!(ImageKind::WebP.mime_type(Path::new("a.webp")), "image/webp");
    }

    #[test]
    fn mime_type_heic() {
        assert_eq!(ImageKind::Sidecar.mime_type(Path::new("a.heic")), "image/heic");
        assert_eq!(ImageKind::Sidecar.mime_type(Path::new("a.heif")), "image/heif");
    }

    #[test]
    fn mime_type_raw_formats() {
        assert_eq!(ImageKind::Sidecar.mime_type(Path::new("a.cr3")), "image/x-canon-cr3");
        assert_eq!(ImageKind::Sidecar.mime_type(Path::new("a.nef")), "image/x-nikon-nef");
        assert_eq!(ImageKind::Sidecar.mime_type(Path::new("a.arw")), "image/x-sony-arw");
        assert_eq!(ImageKind::Sidecar.mime_type(Path::new("a.dng")), "image/x-adobe-dng");
    }

    #[test]
    fn mime_type_fallback() {
        assert_eq!(ImageKind::Jpeg.mime_type(Path::new("noext")), "image/jpeg");
    }

    // ── is_supported_image ───────────────────────────────────────────

    #[test]
    fn supported_image_extensions() {
        assert!(is_supported_image(Path::new("photo.jpg")));
        assert!(is_supported_image(Path::new("photo.JPEG")));
        assert!(is_supported_image(Path::new("photo.png")));
        assert!(is_supported_image(Path::new("photo.webp")));
        assert!(is_supported_image(Path::new("photo.tif")));
        assert!(is_supported_image(Path::new("photo.heic")));
        assert!(is_supported_image(Path::new("photo.cr3")));
        assert!(is_supported_image(Path::new("photo.dng")));
    }

    #[test]
    fn unsupported_image_extensions() {
        assert!(!is_supported_image(Path::new("doc.pdf")));
        assert!(!is_supported_image(Path::new("video.mp4")));
        assert!(!is_supported_image(Path::new("readme.txt")));
        assert!(!is_supported_image(Path::new("noext")));
    }

    // ── collect_images ───────────────────────────────────────────────

    #[test]
    fn collect_images_single_file() {
        let dir = TempDir::new().unwrap();
        let jpg = dir.path().join("test.jpg");
        fs::write(&jpg, b"fake").unwrap();

        let images = collect_images(&[jpg.clone()]);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0], jpg);
    }

    #[test]
    fn collect_images_skips_unsupported() {
        let dir = TempDir::new().unwrap();
        let txt = dir.path().join("readme.txt");
        fs::write(&txt, b"hello").unwrap();

        let images = collect_images(&[txt]);
        assert!(images.is_empty());
    }

    #[test]
    fn collect_images_directory_recursive() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub");
        fs::create_dir(&sub).unwrap();

        fs::write(dir.path().join("a.jpg"), b"fake").unwrap();
        fs::write(sub.join("b.png"), b"fake").unwrap();
        fs::write(sub.join("c.txt"), b"fake").unwrap();

        let images = collect_images(&[dir.path().to_path_buf()]);
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn collect_images_empty_dir() {
        let dir = TempDir::new().unwrap();
        let images = collect_images(&[dir.path().to_path_buf()]);
        assert!(images.is_empty());
    }

    #[test]
    fn collect_images_nonexistent_path() {
        let images = collect_images(&[PathBuf::from("/nonexistent/path")]);
        assert!(images.is_empty());
    }

    #[test]
    fn collect_images_mixed_files_and_dirs() {
        let dir = TempDir::new().unwrap();
        let jpg = dir.path().join("photo.jpg");
        let sub = dir.path().join("folder");
        fs::create_dir(&sub).unwrap();
        fs::write(&jpg, b"fake").unwrap();
        fs::write(sub.join("deep.heic"), b"fake").unwrap();

        let images = collect_images(&[jpg.clone(), sub]);
        assert_eq!(images.len(), 2);
    }

    // ── build_service_chain ──────────────────────────────────────────

    #[test]
    fn build_service_chain_none_enabled() {
        let mut config = Config::default();
        config.ai_services.openai.enabled = false;
        config.ai_services.gemini.enabled = false;
        config.ai_services.cloudflare.enabled = false;

        let services = build_service_chain(&config);
        assert!(services.is_empty());
    }

    #[test]
    fn build_service_chain_skips_empty_keys() {
        let config = Config::default(); // openai enabled but key is empty
        let services = build_service_chain(&config);
        assert!(services.is_empty());
    }

    #[test]
    fn build_service_chain_with_key() {
        let mut config = Config::default();
        config.ai_services.openai.api_key = "sk-test".to_string();

        let services = build_service_chain(&config);
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].name(), "OpenAI");
    }
}
