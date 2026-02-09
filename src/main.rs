mod ai;
mod config;
mod exif;
mod pipeline;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp(None)
        .init();

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

    // Load config
    let mut config = config::Config::load(cli.config.as_deref())?;

    // Override dry_run from CLI flag
    if cli.dry_run {
        config.output.dry_run = true;
    }

    // Validate inputs
    if cli.paths.is_empty() {
        anyhow::bail!("No input files or directories specified. Use --help for usage.");
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
                let action = if config.output.dry_run {
                    "Would write"
                } else {
                    "Wrote"
                };
                log::info!("  {action}: {}", written.join(", "));
            }

            if !result.skipped_fields.is_empty() {
                log::info!("  Skipped: {}", result.skipped_fields.join(", "));
            }

            if let Some(ref service) = result.ai_service_used {
                log::info!("  AI service: {service}");
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
