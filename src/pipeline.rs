use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::ai::{self, AiResult, AiService};
use crate::config::Config;
use crate::exif::{self, ExifData};
use crate::exif::write_exif;

/// Supported image extensions.
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "tif", "tiff"];

/// Result of processing a single image.
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
}

/// Collect image files from the given paths (files or directories).
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

/// Build the AI service chain from config.
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
            _ => {}
        }
    }

    services
}

/// Process a single image through the AI pipeline and write EXIF.
pub async fn process_image(
    path: &Path,
    services: &[Box<dyn AiService>],
    config: &Config,
) -> ProcessResult {
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

    // Build prompt
    let prompt = ai::build_prompt();

    // Try each AI service in order (failover chain)
    let mut errors = Vec::new();
    for service in services {
        log::info!("  Trying {}...", service.name());
        match service.analyze(&image_base64, &prompt).await {
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

    // Write EXIF
    let ai_data = result.ai_result.as_ref().unwrap();
    match write_exif(
        path,
        ai_data,
        &result.existing_exif,
        &config.exif_fields,
        config.output.dry_run,
    ) {
        Ok(write_result) => {
            result.title_written = write_result.title_written;
            result.description_written = write_result.description_written;
            result.tags_written = write_result.tags_written;
            result.gps_written = write_result.gps_written;
            result.subject_written = write_result.subject_written;
            result.skipped_fields = write_result.skipped_fields;
        }
        Err(e) => {
            result.error = Some(format!("Failed to write EXIF: {e}"));
        }
    }

    result
}
