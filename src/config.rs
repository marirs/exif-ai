use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level configuration for the exif-ai library.
///
/// Controls which AI services to use, which metadata fields to write,
/// and output behavior (dry run, backups).
///
/// # Loading
///
/// ```rust,no_run
/// use exif_ai::config::Config;
///
/// // From a JSON file
/// let config = Config::load(Some("config.json".as_ref())).unwrap();
///
/// // Or use defaults and customize
/// let mut config = Config::default();
/// config.ai_services.openai.api_key = "sk-...".into();
/// config.ai_services.openai.enabled = true;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// AI service configurations (OpenAI, Gemini, Cloudflare).
    pub ai_services: AiServices,
    /// Order in which AI services are tried (failover chain).
    pub service_order: Vec<String>,
    /// Which metadata fields to write and overwrite behavior.
    pub exif_fields: ExifFields,
    /// Output behavior (dry run, backups, logging).
    pub output: OutputConfig,
}

/// Configuration for all available AI services.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiServices {
    pub openai: OpenAiConfig,
    pub gemini: GeminiConfig,
    pub cloudflare: CloudflareConfig,
}

/// OpenAI service configuration (GPT-4o-mini, GPT-4o, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
    pub enabled: bool,
}

/// Google Gemini service configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: String,
    pub enabled: bool,
}

/// Cloudflare Workers AI service configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudflareConfig {
    pub account_id: String,
    pub api_token: String,
    pub model: String,
    pub enabled: bool,
}

/// Controls which metadata fields are written to images.
///
/// Each `write_*` flag enables or disables writing that field.
/// `overwrite_existing` controls whether existing values are replaced.
///
/// # Example
///
/// ```rust
/// use exif_ai::config::ExifFields;
///
/// let fields = ExifFields {
///     write_title: true,
///     write_description: true,
///     write_tags: true,
///     write_gps: false,       // don't write GPS
///     write_subject: false,   // don't write subject
///     overwrite_existing: false, // preserve existing values
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExifFields {
    /// Write title (ImageDescription + XPTitle + dc:title).
    pub write_title: bool,
    /// Write description (UserComment + XPComment + dc:description).
    pub write_description: bool,
    /// Write tags/keywords (XPKeywords + dc:subject + IPTC keywords).
    pub write_tags: bool,
    /// Write GPS coordinates (only if image has no existing GPS).
    pub write_gps: bool,
    /// Write subject identification (XPSubject).
    pub write_subject: bool,
    /// If `true`, overwrite existing metadata values. If `false`, skip fields that already have data.
    pub overwrite_existing: bool,
}

/// Output and behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// If `true`, preview what would be written without modifying any files.
    pub dry_run: bool,
    /// If `true`, create a `.bak` backup before modifying an image.
    pub backup_originals: bool,
    /// Optional path to a log file.
    pub log_file: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ai_services: AiServices {
                openai: OpenAiConfig {
                    api_key: String::new(),
                    model: "gpt-4o-mini".to_string(),
                    enabled: true,
                },
                gemini: GeminiConfig {
                    api_key: String::new(),
                    model: "gemini-2.0-flash".to_string(),
                    enabled: false,
                },
                cloudflare: CloudflareConfig {
                    account_id: String::new(),
                    api_token: String::new(),
                    model: "@cf/llava-hf/llava-1.5-7b-hf".to_string(),
                    enabled: false,
                },
            },
            service_order: vec![
                "openai".to_string(),
                "gemini".to_string(),
                "cloudflare".to_string(),
            ],
            exif_fields: ExifFields {
                write_title: true,
                write_description: true,
                write_tags: true,
                write_gps: true,
                write_subject: true,
                overwrite_existing: false,
            },
            output: OutputConfig {
                dry_run: false,
                backup_originals: true,
                log_file: None,
            },
        }
    }
}

impl Config {
    /// Resolve the config file path — same directory as the executable.
    pub fn config_path() -> Result<PathBuf> {
        let exe_path = std::env::current_exe().context("Failed to get executable path")?;
        let exe_dir = exe_path
            .parent()
            .context("Failed to get executable directory")?;
        Ok(exe_dir.join("config.json"))
    }

    /// Load config from the given path, or from the default location.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let config_path = match path {
            Some(p) => p.to_path_buf(),
            None => Self::config_path()?,
        };

        if !config_path.exists() {
            log::warn!(
                "Config file not found at {}. Using defaults.",
                config_path.display()
            );
            return Ok(Self::default());
        }

        let contents =
            std::fs::read_to_string(&config_path).context("Failed to read config file")?;
        let config: Config =
            serde_json::from_str(&contents).context("Failed to parse config file")?;
        Ok(config)
    }

    /// Save config to the given path, or to the default location.
    pub fn save(&self, path: Option<&Path>) -> Result<()> {
        let config_path = match path {
            Some(p) => p.to_path_buf(),
            None => Self::config_path()?,
        };

        let contents = serde_json::to_string_pretty(self).context("Failed to serialize config")?;
        std::fs::write(&config_path, contents).context("Failed to write config file")?;
        log::info!("Config saved to {}", config_path.display());
        Ok(())
    }

    /// Get the ordered list of enabled AI services.
    pub fn enabled_services(&self) -> Vec<String> {
        self.service_order
            .iter()
            .filter(|name| match name.as_str() {
                "openai" => self.ai_services.openai.enabled,
                "gemini" => self.ai_services.gemini.enabled,
                "cloudflare" => self.ai_services.cloudflare.enabled,
                _ => false,
            })
            .cloned()
            .collect()
    }
}
