use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use exif_ai::{config, exif, pipeline};

#[derive(Parser, Debug)]
#[command(
    name = "exif-ai",
    version,
    about = "AI-powered EXIF metadata writer — generate SEO titles, descriptions, tags, GPS, and subject data for images"
)]
struct Cli {
    /// Image files or directories to process
    #[arg(value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Path to config file (default: config.json next to binary)
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Initialize a default config.json and exit
    #[arg(long)]
    init: bool,

    /// Preview changes without writing to files
    #[arg(long)]
    dry_run: bool,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Display all EXIF metadata and exit
    #[arg(long = "show-exif")]
    show_exif: bool,

    /// Clear all EXIF/XMP/IPTC metadata from the image(s)
    #[arg(long = "clear-exif")]
    clear_exif: bool,

    /// Download the local BLIP model for offline inference
    #[arg(long = "download-model")]
    download_model: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp(None)
        .init();

    // Handle --download-model
    if cli.download_model {
        let config = config::Config::load(cli.config.as_deref())?;
        let model_dir = if config.ai_services.local.model_path.is_empty() {
            None
        } else {
            Some(std::path::Path::new(&config.ai_services.local.model_path))
        };
        println!("Downloading BLIP model...");
        let dir = exif_ai::ai::local::download_model(model_dir).await?;
        println!("Model downloaded to: {}", dir.display());
        println!("\nTo enable local inference, set \"local.enabled\": true in your config.json");
        return Ok(());
    }

    // Handle --init
    if cli.init {
        let config = config::Config::default();
        let path = cli.config.as_deref();
        config.save(path)?;
        let save_path = match path {
            Some(p) => p.to_path_buf(),
            None => config::Config::config_path()?,
        };
        println!("Default config written to {}", save_path.display());
        return Ok(());
    }

    // Validate inputs for non-init commands
    if !cli.init && cli.paths.is_empty() {
        anyhow::bail!("No input files or directories specified. Use --help for usage.");
    }

    // Handle --show-exif
    if cli.show_exif {
        let images = pipeline::collect_images(&cli.paths);
        if images.is_empty() {
            anyhow::bail!("No supported image files found in the specified paths.");
        }
        for image_path in &images {
            print_full_exif(image_path)?;
        }
        return Ok(());
    }

    // Handle --clear-exif
    if cli.clear_exif {
        let images = pipeline::collect_images(&cli.paths);
        if images.is_empty() {
            anyhow::bail!("No supported image files found in the specified paths.");
        }
        for image_path in &images {
            let kind = pipeline::ImageKind::from_path(image_path);
            match kind {
                Some(k) => {
                    match exif::clear_exif(image_path, k) {
                        Ok(()) => log::info!("Cleared EXIF: {}", image_path.display()),
                        Err(e) => log::error!("Failed to clear {}: {e}", image_path.display()),
                    }
                }
                None => log::warn!("Unsupported format: {}", image_path.display()),
            }
        }
        return Ok(());
    }

    // Load config
    let mut config = config::Config::load(cli.config.as_deref())?;

    // Override dry_run from CLI flag
    if cli.dry_run {
        config.output.dry_run = true;
    }

    // Collect images
    let images = pipeline::collect_images(&cli.paths);
    if images.is_empty() {
        anyhow::bail!("No supported image files found in the specified paths.");
    }

    log::info!("Found {} image(s) to process", images.len());
    if config.output.dry_run {
        log::info!("DRY RUN — no files will be modified");
    }

    // Build AI service chain
    let services = pipeline::build_service_chain(&config);
    if services.is_empty() {
        anyhow::bail!(
            "No AI services configured. Run `exif-ai --init` to create a config file, then add your API keys."
        );
    }

    log::info!(
        "AI chain: {}",
        config
            .enabled_services()
            .join(" → ")
    );

    // Process each image
    let mut results = Vec::new();
    let total = images.len();

    for (i, image_path) in images.iter().enumerate() {
        log::info!(
            "[{}/{}] Processing: {}",
            i + 1,
            total,
            image_path.display()
        );

        let result = pipeline::process_image(image_path, &services, &config).await;

        // Print result
        if let Some(ref err) = result.error {
            log::error!("  Error: {err}");
        } else {
            if let Some(ref service) = result.ai_service_used {
                log::info!("  AI service: {service}");
            }

            // Show EXIF preview table
            if config.output.dry_run {
                print_exif_preview(&result);
            } else {
                let mut written = Vec::new();
                if result.title_written {
                    written.push("title");
                }
                if result.description_written {
                    written.push("description");
                }
                if result.tags_written {
                    written.push("tags");
                }
                if result.gps_written {
                    written.push("gps");
                }
                if result.subject_written {
                    written.push("subject");
                }

                if !written.is_empty() {
                    log::info!("  Wrote: {}", written.join(", "));
                }

                if let Some(ref sidecar) = result.sidecar_path {
                    log::info!("  Sidecar XMP: {}", sidecar.display());
                }

                if !result.skipped_fields.is_empty() {
                    log::info!("  Skipped: {}", result.skipped_fields.join(", "));
                }
            }
        }

        results.push(result);
    }

    // JSON output
    if cli.json {
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "path": r.path.display().to_string(),
                    "ai_service": r.ai_service_used,
                    "ai_result": r.ai_result,
                    "title_written": r.title_written,
                    "description_written": r.description_written,
                    "tags_written": r.tags_written,
                    "gps_written": r.gps_written,
                    "subject_written": r.subject_written,
                    "skipped_fields": r.skipped_fields,
                    "sidecar_path": r.sidecar_path.as_ref().map(|p| p.display().to_string()),
                    "error": r.error,
                })
            })
            .collect();

        println!("{}", serde_json::to_string_pretty(&json_results)?);
    }

    // Summary
    let success = results.iter().filter(|r| r.error.is_none()).count();
    let failed = results.iter().filter(|r| r.error.is_some()).count();
    log::info!("Done: {success} succeeded, {failed} failed out of {total} images");

    Ok(())
}

// ANSI color codes
const GREEN: &str = "\x1b[32m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

/// Print an EXIF preview table showing existing data and new AI values for dry-run mode.
fn print_exif_preview(result: &pipeline::ProcessResult) {
    let existing = &result.existing_exif;
    let ai = match &result.ai_result {
        Some(ai) => ai,
        None => return,
    };

    println!();
    println!("  {BOLD}EXIF Data:{RESET}");
    println!("  {DIM}{}{RESET}", "─".repeat(72));

    // --- Existing standard EXIF fields ---
    print_existing("Make", existing.make.as_deref());
    print_existing("Model", existing.model.as_deref());
    print_existing("LensModel", existing.lens_model.as_deref());
    print_existing("DateTimeOriginal", existing.date_time.as_deref());
    print_existing("ExposureTime", existing.exposure_time.as_deref());
    print_existing("FNumber", existing.f_number.as_deref());
    print_existing("ISO", existing.iso.as_deref());
    print_existing("FocalLength", existing.focal_length.as_deref());
    print_existing("ColorSpace", existing.color_space.as_deref());
    print_existing("Orientation", existing.orientation.as_deref());
    print_existing("Software", existing.software.as_deref());

    // Image dimensions
    if let (Some(w), Some(h)) = (&existing.image_width, &existing.image_height) {
        print_existing_val("ImageSize", &format!("{w} x {h}"));
    }

    print_existing("XResolution", existing.x_resolution.as_deref());
    print_existing("YResolution", existing.y_resolution.as_deref());

    // GPS (existing)
    if existing.has_gps {
        let lat = existing.gps_latitude.unwrap_or(0.0);
        let lon = existing.gps_longitude.unwrap_or(0.0);
        print_existing_val("GPSLatitude", &format!("{lat:.6}"));
        print_existing_val("GPSLongitude", &format!("{lon:.6}"));
    }

    // --- Existing writable fields (may be overwritten) ---
    print_existing("ImageDescription", existing.title.as_deref());
    print_existing("UserComment", existing.description.as_deref());
    print_existing("XPKeywords", existing.keywords.as_deref());
    print_existing("XPSubject", existing.subject.as_deref());

    // --- Separator before new AI values ---
    println!("  {DIM}{}{RESET}", "─".repeat(72));
    println!("  {BOLD}New (AI-generated):{RESET}");
    println!("  {DIM}{}{RESET}", "─".repeat(72));

    // Title
    if let Some(ref title) = ai.title {
        if result.title_written {
            print_new("ImageDescription", title);
            print_new("XPTitle", title);
        } else {
            print_skipped("ImageDescription", "(exists, skipped)");
        }
    }

    // Description
    if let Some(ref desc) = ai.description {
        if result.description_written {
            print_new("UserComment", desc);
            print_new("XPComment", desc);
        } else {
            print_skipped("UserComment", "(exists, skipped)");
        }
    }

    // Tags
    if let Some(ref tags) = ai.tags {
        let kw = tags.join("; ");
        if result.tags_written {
            print_new("XPKeywords", &kw);
        } else {
            print_skipped("XPKeywords", "(exists, skipped)");
        }
    }

    // Subject
    if let Some(ref subjects) = ai.subject {
        if !subjects.is_empty() {
            let subj = subjects.join("; ");
            if result.subject_written {
                print_new("XPSubject", &subj);
            } else {
                print_skipped("XPSubject", "(exists, skipped)");
            }
        }
    }

    // GPS (new)
    if let Some(ref gps) = ai.gps {
        if result.gps_written {
            print_new("GPSLatitude", &format!("{:.6}", gps.latitude));
            print_new("GPSLongitude", &format!("{:.6}", gps.longitude));
        } else if existing.has_gps {
            print_skipped("GPS", "(exists, skipped)");
        }
    }

    println!("  {DIM}{}{RESET}", "─".repeat(72));
    println!("  {GREEN}*{RESET} = new value to be written");
    println!();
}

/// Max width for the value column before wrapping.
const VAL_WIDTH: usize = 46;
/// Indent for continuation lines (tag column width + " : " = 25 chars + 2 leading spaces).
const INDENT: &str = "                           ";

/// Print an existing EXIF field row.
fn print_existing(tag: &str, value: Option<&str>) {
    if let Some(val) = value {
        if !val.is_empty() {
            print_existing_val(tag, val);
        }
    }
}

/// Print an existing EXIF value row.
fn print_existing_val(tag: &str, val: &str) {
    let tag_col = format!("{:<22}", tag);
    let lines = wrap_text(val, VAL_WIDTH);
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            println!("  {tag_col} : {line}");
        } else {
            println!("  {INDENT}{line}");
        }
    }
}

/// Print a new AI-generated value row (green with *).
fn print_new(tag: &str, val: &str) {
    let tag_col = format!("{:<22}", tag);
    let lines = wrap_text(val, VAL_WIDTH);
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            if lines.len() == 1 {
                println!("  {GREEN}{tag_col} : {line} *{RESET}");
            } else {
                println!("  {GREEN}{tag_col} : {line}{RESET}");
            }
        } else if i == lines.len() - 1 {
            println!("  {GREEN}{INDENT}{line} *{RESET}");
        } else {
            println!("  {GREEN}{INDENT}{line}{RESET}");
        }
    }
}

/// Print a skipped field row (dimmed).
fn print_skipped(tag: &str, reason: &str) {
    let tag_col = format!("{:<22}", tag);
    println!("  {DIM}{tag_col} : {reason}{RESET}");
}

/// Print full EXIF metadata for a file, organized by section.
fn print_full_exif(path: &std::path::Path) -> Result<()> {
    let data = exif::read_exif(path)?;

    println!();
    println!("{BOLD}File:{RESET} {}", path.display());
    println!("{DIM}{}{RESET}", "═".repeat(72));

    // --- Camera / Device ---
    let camera_fields: Vec<(&str, Option<&str>)> = vec![
        ("Make", data.make.as_deref()),
        ("Model", data.model.as_deref()),
        ("LensModel", data.lens_model.as_deref()),
        ("Software", data.software.as_deref()),
    ];
    if camera_fields.iter().any(|(_, v)| v.is_some()) {
        println!("  {BOLD}Camera / Device{RESET}");
        println!("  {DIM}{}{RESET}", "─".repeat(70));
        for (tag, val) in &camera_fields {
            if let Some(v) = val {
                print_row(tag, v);
            }
        }
        println!();
    }

    // --- Capture Settings ---
    let capture_fields: Vec<(&str, Option<&str>)> = vec![
        ("DateTimeOriginal", data.date_time.as_deref()),
        ("ExposureTime", data.exposure_time.as_deref()),
        ("FNumber", data.f_number.as_deref()),
        ("ISO", data.iso.as_deref()),
        ("FocalLength", data.focal_length.as_deref()),
        ("Orientation", data.orientation.as_deref()),
    ];
    if capture_fields.iter().any(|(_, v)| v.is_some()) {
        println!("  {BOLD}Capture Settings{RESET}");
        println!("  {DIM}{}{RESET}", "─".repeat(70));
        for (tag, val) in &capture_fields {
            if let Some(v) = val {
                print_row(tag, v);
            }
        }
        println!();
    }

    // --- Image Properties ---
    let mut image_fields: Vec<(&str, Option<String>)> = vec![
        ("ColorSpace", data.color_space.clone()),
        ("XResolution", data.x_resolution.clone()),
        ("YResolution", data.y_resolution.clone()),
    ];
    if let (Some(w), Some(h)) = (&data.image_width, &data.image_height) {
        image_fields.insert(0, ("ImageSize", Some(format!("{w} x {h}"))));
    }
    if image_fields.iter().any(|(_, v)| v.is_some()) {
        println!("  {BOLD}Image Properties{RESET}");
        println!("  {DIM}{}{RESET}", "─".repeat(70));
        for (tag, val) in &image_fields {
            if let Some(v) = val {
                print_row(tag, v);
            }
        }
        println!();
    }

    // --- GPS ---
    if data.has_gps {
        println!("  {BOLD}GPS{RESET}");
        println!("  {DIM}{}{RESET}", "─".repeat(70));
        if let Some(lat) = data.gps_latitude {
            print_row("GPSLatitude", &format!("{lat:.6}"));
        }
        if let Some(lon) = data.gps_longitude {
            print_row("GPSLongitude", &format!("{lon:.6}"));
        }
        println!();
    }

    // --- AI / Descriptive Metadata ---
    let desc_fields: Vec<(&str, Option<&str>)> = vec![
        ("ImageDescription", data.title.as_deref()),
        ("UserComment", data.description.as_deref()),
        ("XPKeywords", data.keywords.as_deref()),
        ("XPSubject", data.subject.as_deref()),
    ];
    if desc_fields.iter().any(|(_, v)| v.is_some()) {
        println!("  {BOLD}Descriptive Metadata{RESET}");
        println!("  {DIM}{}{RESET}", "─".repeat(70));
        for (tag, val) in &desc_fields {
            if let Some(v) = val {
                print_row(tag, v);
            }
        }
        println!();
    }

    // If completely empty
    let has_any = data.make.is_some()
        || data.model.is_some()
        || data.date_time.is_some()
        || data.exposure_time.is_some()
        || data.image_width.is_some()
        || data.has_gps
        || data.title.is_some()
        || data.description.is_some()
        || data.keywords.is_some()
        || data.software.is_some();
    if !has_any {
        println!("  {DIM}(no EXIF metadata found){RESET}");
        println!();
    }

    Ok(())
}

/// Print a single row in the EXIF display table.
fn print_row(tag: &str, val: &str) {
    let tag_col = format!("{:<22}", tag);
    let lines = wrap_text(val, VAL_WIDTH);
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            println!("  {tag_col} : {line}");
        } else {
            println!("  {INDENT}{line}");
        }
    }
}

/// Wrap text at word boundaries to fit within max_width.
fn wrap_text(s: &str, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in s.split_whitespace() {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + 1 + word.len() <= max_width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(s.to_string());
    }

    lines
}
