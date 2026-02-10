#[cfg(target_os = "macos")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::blip;
use tokenizers::Tokenizer;

use std::path::{Path, PathBuf};

use super::{AiResult, AiService};

/// Default model directory name inside the user's cache.
const MODEL_DIR_NAME: &str = "exif-ai";

/// BLIP model filename (safetensors — supports Metal GPU).
const MODEL_FILENAME: &str = "model.safetensors";

/// Tokenizer filename.
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// HuggingFace repo for the BLIP model (safetensors format).
const MODEL_REPO: &str = "Salesforce/blip-image-captioning-large";

/// HuggingFace revision that has the safetensors file.
const MODEL_REVISION: &str = "refs/pr/18";

/// SEP token ID used by BLIP to signal end of generation.
const SEP_TOKEN_ID: u32 = 102;

/// BOS token ID used by BLIP to start generation.
const BOS_TOKEN_ID: u32 = 30522;

/// Select the best available device (Metal GPU on macOS, CPU elsewhere).
fn best_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                log::info!("Using Metal GPU for inference");
                return Ok(device);
            }
            Err(e) => {
                log::warn!("Metal GPU not available ({e}), falling back to CPU");
            }
        }
    }
    log::info!("Using CPU for inference");
    Ok(Device::Cpu)
}

/// Local AI service using a BLIP model for image captioning.
///
/// Runs entirely on-device with no network calls. Uses Metal GPU on macOS
/// for fast inference (~2-5s), or CPU with Accelerate BLAS on Apple Silicon.
/// The model must be downloaded first using [`download_model`] or
/// `--download-model` CLI flag.
pub struct LocalService {
    model_path: PathBuf,
    tokenizer_path: PathBuf,
}

impl LocalService {
    /// Create a new LocalService with explicit model and tokenizer paths.
    pub fn new(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            model_path,
            tokenizer_path,
        }
    }

    /// Create a LocalService using the default model directory.
    pub fn from_default_dir() -> Result<Self> {
        let dir = default_model_dir()?;
        let model_path = dir.join(MODEL_FILENAME);
        let tokenizer_path = dir.join(TOKENIZER_FILENAME);
        Ok(Self {
            model_path,
            tokenizer_path,
        })
    }

    /// Check if the model files exist.
    pub fn model_exists(&self) -> bool {
        self.model_path.exists() && self.tokenizer_path.exists()
    }

    /// Generate a caption for an image file.
    ///
    /// Tries Metal GPU first on macOS; if that fails (e.g. missing kernels),
    /// falls back to CPU with Accelerate BLAS.
    fn caption_image(&self, image_path: &Path) -> Result<String> {
        let device = best_device()?;

        match self.run_inference(image_path, &device) {
            Ok(caption) => Ok(caption),
            Err(e) if !matches!(device, Device::Cpu) => {
                log::warn!("Inference failed on GPU ({e}), retrying on CPU...");
                self.run_inference(image_path, &Device::Cpu)
            }
            Err(e) => Err(e),
        }
    }

    /// Run BLIP inference on the given device.
    fn run_inference(&self, image_path: &Path, device: &Device) -> Result<String> {
        // Load and preprocess image
        let image = load_image(image_path, device)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&self.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Load BLIP model from safetensors
        let config = blip::Config::image_captioning_large();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&self.model_path], DType::F32, device)?
        };
        let mut model = blip::BlipForConditionalGeneration::new(&config, vb)?;

        // Encode image through vision model
        let image_embeds = image
            .unsqueeze(0)?
            .apply(model.vision_model())?;

        // Generate caption tokens autoregressively
        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::new(1337, None, None);
        let mut token_ids = vec![BOS_TOKEN_ID];

        for index in 0..1000 {
            let context_size = if index > 0 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], device)?.unsqueeze(0)?;
            let logits = model.text_decoder().forward(&input_ids, &image_embeds)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let logits = logits.to_device(&Device::Cpu)?;
            let token = logits_processor.sample(&logits)?;
            if token == SEP_TOKEN_ID {
                break;
            }
            token_ids.push(token);
        }

        // Decode tokens to text
        let caption = tokenizer
            .decode(&token_ids[1..], true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {e}"))?;

        Ok(caption.trim().to_string())
    }
}

#[async_trait::async_trait]
impl AiService for LocalService {
    fn name(&self) -> &str {
        "Local (BLIP)"
    }

    async fn analyze(
        &self,
        _image_base64: &str,
        _prompt: &str,
        _mime_type: &str,
    ) -> Result<AiResult> {
        anyhow::bail!(
            "LocalService does not support base64 analysis. Use analyze_file() instead."
        )
    }

    fn supports_file_analysis(&self) -> bool {
        true
    }

    fn analyze_file(&self, image_path: &std::path::Path) -> Result<AiResult> {
        if !self.model_exists() {
            anyhow::bail!(
                "Local model not found. Run `exif-ai-cli --download-model` to download it."
            );
        }

        log::info!("Running local BLIP inference on: {}", image_path.display());
        let caption = self.caption_image(image_path)?;
        log::info!("Caption: {caption}");

        // Build structured result from caption
        let title = build_title(&caption);
        let tags = extract_tags(&caption);

        Ok(AiResult {
            title: Some(title),
            description: Some(caption),
            tags: if tags.is_empty() { None } else { Some(tags) },
            gps: None,
            subject: None,
        })
    }
}

/// Load and preprocess an image for BLIP (resize to 384×384, normalize).
fn load_image(path: &Path, device: &Device) -> Result<Tensor> {
    let img = image::ImageReader::open(path)
        .context("Failed to open image")?
        .decode()
        .context("Failed to decode image")?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (384, 384, 3), device)?
        .permute((2, 0, 1))?;
    // OpenAI CLIP normalization
    let mean =
        Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], device)?.reshape((3, 1, 1))?;
    let std =
        Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], device)?.reshape((3, 1, 1))?;
    let normalized = (data.to_dtype(DType::F32)? / 255.0)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?;
    Ok(normalized)
}

/// Build a short title from a caption by extracting key noun phrases.
///
/// Strips filler words and prepositions to produce a concise title
/// distinct from the full description, e.g.
/// "trees and leaves on the ground in a park with a fence" → "Trees and Leaves in Park"
fn build_title(caption: &str) -> String {
    let strip_words: &[&str] = &[
        "a", "an", "the", "on", "in", "at", "of", "with", "by", "from",
        "to", "for", "is", "are", "was", "were", "that", "this", "it",
        "its", "some", "very", "just",
    ];

    let words: Vec<&str> = caption.split_whitespace().collect();

    // Keep meaningful words, limit to ~6 words for a concise title
    let keep_lower: &[&str] = &["and", "or", "but", "nor"];
    let meaningful: Vec<&str> = words
        .iter()
        .filter(|w| {
            let lower = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            lower.len() > 1 && !strip_words.contains(&lower.as_str())
        })
        .copied()
        .collect();

    let title_words: Vec<String> = meaningful
        .iter()
        .take(6)
        .enumerate()
        .map(|(i, w)| {
            let trimmed = w.trim_matches(|c: char| !c.is_alphanumeric());
            let lower = trimmed.to_lowercase();
            // Keep conjunctions lowercase (except first word)
            if i > 0 && keep_lower.contains(&lower.as_str()) {
                return lower;
            }
            // Title-case
            let mut chars = trimmed.chars();
            match chars.next() {
                Some(c) => {
                    let mut s = c.to_uppercase().to_string();
                    s.extend(chars);
                    s
                }
                None => String::new(),
            }
        })
        .collect();

    if title_words.is_empty() {
        // Fallback: capitalize the first few words of the caption
        let mut title = String::new();
        let mut chars = caption.chars();
        if let Some(first) = chars.next() {
            title.extend(first.to_uppercase());
            title.extend(chars);
        }
        if title.len() > 50 {
            if let Some(pos) = title[..50].rfind(' ') {
                title.truncate(pos);
            }
        }
        return title;
    }

    title_words.join(" ")
}

/// Extract keyword tags from a caption using simple NLP heuristics.
fn extract_tags(caption: &str) -> Vec<String> {
    let stop_words: &[&str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "that", "this", "it",
        "its", "which", "what", "who", "whom", "their", "them", "they", "he",
        "she", "his", "her", "we", "you", "your", "my", "me", "up", "down",
    ];

    caption
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| w.len() > 2 && !stop_words.contains(&w.as_str()))
        .collect::<Vec<_>>()
        .into_iter()
        .fold(Vec::new(), |mut acc, w| {
            if !acc.contains(&w) {
                acc.push(w);
            }
            acc
        })
        .into_iter()
        .take(10)
        .collect()
}

/// Get the default model directory path.
pub fn default_model_dir() -> Result<PathBuf> {
    let cache_dir = dirs_cache_dir().context("Could not determine cache directory")?;
    Ok(cache_dir.join(MODEL_DIR_NAME))
}

/// Platform-specific cache directory.
fn dirs_cache_dir() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Caches"))
    }
    #[cfg(target_os = "linux")]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache"))
    }
}

/// Download the BLIP model and tokenizer to the specified directory.
///
/// If `model_dir` is `None`, uses the default cache directory.
pub async fn download_model(model_dir: Option<&Path>) -> Result<PathBuf> {
    let dir = match model_dir {
        Some(d) => d.to_path_buf(),
        None => default_model_dir()?,
    };

    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create model directory: {}", dir.display()))?;

    let model_dest = dir.join(MODEL_FILENAME);
    let tokenizer_dest = dir.join(TOKENIZER_FILENAME);

    let api = hf_hub::api::tokio::Api::new()?;

    // Download BLIP model (safetensors format — supports Metal GPU)
    if model_dest.exists() {
        log::info!("Model already exists: {}", model_dest.display());
    } else {
        log::info!("Downloading BLIP model from {} ({})...", MODEL_REPO, MODEL_REVISION);
        let repo = api.repo(hf_hub::Repo::with_revision(
            MODEL_REPO.to_string(),
            hf_hub::RepoType::Model,
            MODEL_REVISION.to_string(),
        ));
        let downloaded = repo.get(MODEL_FILENAME).await
            .context("Failed to download BLIP model")?;
        std::fs::copy(&downloaded, &model_dest)
            .context("Failed to copy model to destination")?;
        log::info!("Model saved to: {}", model_dest.display());
    }

    // Download tokenizer (from main branch)
    if tokenizer_dest.exists() {
        log::info!("Tokenizer already exists: {}", tokenizer_dest.display());
    } else {
        log::info!("Downloading tokenizer from {}...", MODEL_REPO);
        let repo = api.model(MODEL_REPO.to_string());
        let downloaded = repo.get(TOKENIZER_FILENAME).await
            .context("Failed to download tokenizer")?;
        std::fs::copy(&downloaded, &tokenizer_dest)
            .context("Failed to copy tokenizer to destination")?;
        log::info!("Tokenizer saved to: {}", tokenizer_dest.display());
    }

    log::info!("Model ready at: {}", dir.display());
    Ok(dir)
}
