//! # exif-ai
//!
//! AI-powered EXIF metadata writer — generate SEO titles, descriptions, tags, GPS coordinates,
//! and subject data for images using AI vision models (OpenAI, Google Gemini, Cloudflare Workers AI).
//!
//! ## Quick Start
//!
//! The simplest way to use the library is through the pipeline module, which handles
//! the full read → AI analyze → write flow:
//!
//! ```rust,no_run
//! use exif_ai::config::Config;
//! use exif_ai::pipeline::{build_service_chain, collect_images, process_image};
//! use std::path::PathBuf;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Load config from file (contains API keys, field settings, etc.)
//!     let config = Config::load(Some("config.json".as_ref()))?;
//!
//!     // Build the AI service failover chain from config
//!     let services = build_service_chain(&config);
//!
//!     // Collect supported image files from paths (files or directories)
//!     let images = collect_images(&[PathBuf::from("./photos")]);
//!
//!     for path in &images {
//!         let result = process_image(path, &services, &config).await;
//!
//!         if let Some(ref err) = result.error {
//!             eprintln!("Error processing {}: {err}", path.display());
//!         } else {
//!             println!("Processed: {}", path.display());
//!             if let Some(ref sidecar) = result.sidecar_path {
//!                 println!("  Sidecar XMP written: {}", sidecar.display());
//!             }
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Lower-Level Usage
//!
//! For more control, you can call the EXIF reader, AI services, and writer individually:
//!
//! ```rust,no_run
//! use exif_ai::exif::{read_exif, write_exif};
//! use exif_ai::ai::{OpenAiService, AiService, build_prompt};
//! use exif_ai::config::ExifFields;
//! use exif_ai::pipeline::ImageKind;
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let path = Path::new("photo.jpg");
//!
//!     // 1. Read existing EXIF metadata
//!     let existing = read_exif(path)?;
//!     println!("Camera: {:?}", existing.model);
//!
//!     // 2. Analyze with AI
//!     let service = OpenAiService::new("sk-...".into(), "gpt-4o-mini".into());
//!     let bytes = std::fs::read(path)?;
//!     let b64 = base64::Engine::encode(
//!         &base64::engine::general_purpose::STANDARD, &bytes,
//!     );
//!     let ai_result = service.analyze(&b64, &build_prompt(), "image/jpeg").await?;
//!     println!("AI title: {:?}", ai_result.title);
//!
//!     // 3. Write metadata back (format-aware)
//!     let fields = ExifFields {
//!         write_title: true,
//!         write_description: true,
//!         write_tags: true,
//!         write_gps: true,
//!         write_subject: true,
//!         overwrite_existing: false,
//!     };
//!     let result = write_exif(path, &ai_result, &existing, &fields, false, ImageKind::Jpeg)?;
//!     println!("Title written: {}", result.title_written);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Supported Formats
//!
//! | Format | Write Strategy |
//! |--------|---------------|
//! | JPEG (`.jpg`, `.jpeg`) | Native — EXIF + XMP + IPTC |
//! | PNG (`.png`) | Native — XMP in iTXt chunk |
//! | WebP (`.webp`) | Native — EXIF + XMP in RIFF |
//! | TIFF (`.tif`, `.tiff`) | Native — EXIF |
//! | HEIC/HEIF (`.heic`, `.heif`) | Sidecar `.xmp` file |
//! | AVIF (`.avif`) | Sidecar `.xmp` file |
//! | RAW (`.cr2`, `.cr3`, `.dng`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.pef`, `.srw`) | Sidecar `.xmp` file |
//!
//! ## Modules
//!
//! - [`ai`] — AI service trait and implementations (OpenAI, Gemini, Cloudflare)
//! - [`config`] — Configuration types and loading/saving
//! - [`exif`] — EXIF/XMP/IPTC reading and writing
//! - [`pipeline`] — High-level processing pipeline, image collection, and format detection

pub mod ai;
pub mod config;
pub mod exif;
pub mod pipeline;
