use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ai_services: AiServices,
    pub service_order: Vec<String>,
    pub exif_fields: ExifFields,
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiServices {
    pub openai: OpenAiConfig,
    pub gemini: GeminiConfig,
    pub cloudflare: CloudflareConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudflareConfig {
    pub account_id: String,
    pub api_token: String,
    pub model: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExifFields {
    pub write_title: bool,
    pub write_description: bool,
    pub write_tags: bool,
    pub write_gps: bool,
    pub write_subject: bool,
    pub overwrite_existing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub dry_run: bool,
    pub backup_originals: bool,
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
