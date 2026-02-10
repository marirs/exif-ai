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
    #[serde(default)]
    pub local: LocalConfig,
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

/// Local BLIP model configuration.
///
/// When enabled, runs a BLIP image-captioning model on-device.
/// No API keys or network access required after the initial model download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConfig {
    /// Path to the directory containing the model and tokenizer files.
    /// If empty, uses the default cache directory (~/.cache/exif-ai or platform equivalent).
    pub model_path: String,
    pub enabled: bool,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            enabled: false,
        }
    }
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
                local: LocalConfig::default(),
            },
            service_order: vec![
                "local".to_string(),
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
                "local" => self.ai_services.local.enabled,
                _ => false,
            })
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── Config::default ──────────────────────────────────────────────

    #[test]
    fn default_config_has_sane_values() {
        let config = Config::default();
        assert!(config.ai_services.openai.enabled);
        assert!(!config.ai_services.gemini.enabled);
        assert!(!config.ai_services.cloudflare.enabled);
        assert!(config.ai_services.openai.api_key.is_empty());
        assert_eq!(config.ai_services.openai.model, "gpt-4o-mini");
        assert!(!config.ai_services.local.enabled);
        assert!(config.ai_services.local.model_path.is_empty());
        assert_eq!(config.service_order, vec!["local", "openai", "gemini", "cloudflare"]);
    }

    #[test]
    fn default_exif_fields() {
        let config = Config::default();
        assert!(config.exif_fields.write_title);
        assert!(config.exif_fields.write_description);
        assert!(config.exif_fields.write_tags);
        assert!(config.exif_fields.write_gps);
        assert!(config.exif_fields.write_subject);
        assert!(!config.exif_fields.overwrite_existing);
    }

    #[test]
    fn default_output_config() {
        let config = Config::default();
        assert!(!config.output.dry_run);
        assert!(config.output.backup_originals);
        assert!(config.output.log_file.is_none());
    }

    // ── Config::save / Config::load round-trip ───────────────────────

    #[test]
    fn save_and_load_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_config.json");

        let mut config = Config::default();
        config.ai_services.openai.api_key = "sk-test-key".to_string();
        config.ai_services.gemini.enabled = true;
        config.exif_fields.overwrite_existing = true;
        config.output.dry_run = true;

        config.save(Some(&path)).unwrap();
        assert!(path.exists());

        let loaded = Config::load(Some(&path)).unwrap();
        assert_eq!(loaded.ai_services.openai.api_key, "sk-test-key");
        assert!(loaded.ai_services.gemini.enabled);
        assert!(loaded.exif_fields.overwrite_existing);
        assert!(loaded.output.dry_run);
    }

    #[test]
    fn load_nonexistent_returns_default() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("does_not_exist.json");

        let config = Config::load(Some(&path)).unwrap();
        // Should return defaults when file doesn't exist
        assert!(config.ai_services.openai.enabled);
        assert!(config.ai_services.openai.api_key.is_empty());
    }

    #[test]
    fn load_invalid_json_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "not valid json {{{").unwrap();

        let result = Config::load(Some(&path));
        assert!(result.is_err());
    }

    // ── Config::enabled_services ─────────────────────────────────────

    #[test]
    fn enabled_services_default() {
        let config = Config::default();
        let enabled = config.enabled_services();
        // Only openai is enabled by default
        assert_eq!(enabled, vec!["openai"]);
    }

    #[test]
    fn enabled_services_all() {
        let mut config = Config::default();
        config.ai_services.gemini.enabled = true;
        config.ai_services.cloudflare.enabled = true;

        let enabled = config.enabled_services();
        assert_eq!(enabled, vec!["openai", "gemini", "cloudflare"]);
    }

    #[test]
    fn enabled_services_none() {
        let mut config = Config::default();
        config.ai_services.openai.enabled = false;

        let enabled = config.enabled_services();
        assert!(enabled.is_empty());
    }

    // ── Serialization ────────────────────────────────────────────────

    #[test]
    fn config_serializes_to_json() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("openai"));
        assert!(json.contains("service_order"));
        assert!(json.contains("exif_fields"));
    }

    #[test]
    fn config_deserializes_from_json() {
        let json = r#"{
            "ai_services": {
                "openai": {"api_key": "key1", "model": "gpt-4o", "enabled": true},
                "gemini": {"api_key": "", "model": "gemini-2.0-flash", "enabled": false},
                "cloudflare": {"account_id": "", "api_token": "", "model": "test", "enabled": false}
            },
            "service_order": ["openai"],
            "exif_fields": {
                "write_title": true,
                "write_description": false,
                "write_tags": true,
                "write_gps": false,
                "write_subject": false,
                "overwrite_existing": true
            },
            "output": {
                "dry_run": true,
                "backup_originals": false,
                "log_file": null
            }
        }"#;

        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.ai_services.openai.api_key, "key1");
        assert_eq!(config.ai_services.openai.model, "gpt-4o");
        assert!(!config.exif_fields.write_description);
        assert!(config.exif_fields.overwrite_existing);
        assert!(config.output.dry_run);
        assert!(!config.output.backup_originals);
    }
}
