# exif-ai

[![CI](https://github.com/marirs/exif-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/marirs/exif-ai/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/exif-ai.svg)](https://crates.io/crates/exif-ai)

AI-powered EXIF metadata writer — generate SEO titles, descriptions, tags, GPS, and subject data for images using AI vision models.

**Platforms:** Linux (x86_64, ARM64) · macOS (Apple Silicon, Intel) · Windows (x86_64, ARM64)

## Features

- **AI Vision Analysis** — Send images to AI models for intelligent metadata generation
- **Local AI (Offline)** — Run a BLIP model on-device — no API keys, no network, fully private
- **Multi-Service Failover** — Configurable chain: Local BLIP → OpenAI GPT-4o-mini → Google Gemini → Cloudflare Workers AI
- **Multi-Format Support** — JPEG, PNG, WebP, TIFF (native write), HEIC/HEIF, AVIF, and 10+ RAW formats (sidecar XMP)
- **EXIF Writing** — Writes title, description, tags, GPS coordinates, and subject identification directly into image EXIF data
- **GPS Intelligence** — Only writes GPS coordinates when the image has no existing GPS data AND the AI identifies a known location
- **Subject Detection** — Identifies known people, bird species, animal species, and landmarks
- **Batch Processing** — Process single files, multiple files, or entire directories recursively
- **Backup Originals** — Automatically backs up images before modifying EXIF data
- **Dry Run Mode** — Preview what would be written without modifying any files
- **JSON Output** — Machine-readable output for scripting and automation
- **Desktop GUI** — Native desktop app (egui) with image preview, drag-and-drop, EXIF viewer, and AI results
- **Cross-Platform** — Builds for macOS (Silicon & Intel) and Windows (ARM & Intel)
- **Show & Clear EXIF** — Inspect or strip all EXIF/XMP/IPTC metadata from images

## Installation

### As a Library

```toml
[dependencies]
exif-ai = { version = "0.2", default-features = false }
```

### CLI Binary (from source)

```bash
git clone https://github.com/marirs/exif-ai.git
cd exif-ai
cargo build --release
```

The binary will be at `target/release/exif-ai-cli`.

### Desktop GUI (from source)

```bash
cargo build --release --features gui
```

The binary will be at `target/release/exif-ai-gui`.

#### macOS `.app` Bundle

To build a native macOS app bundle (double-click to launch, no terminal):

```bash
cargo install cargo-bundle   # one-time
cargo bundle --release --features gui --bin exif-ai-gui
```

The app will be at `target/release/bundle/osx/Exif AI.app`. Copy it to `/Applications` to install.

#### Windows

```bash
cargo build --release --features gui
```

The `.exe` at `target/release/exif-ai-gui.exe` has the app icon embedded and launches without a console window.

#### Linux

```bash
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libgtk-3-dev
cargo build --release --features gui
```

To integrate with the desktop, copy the binary, icon, and `.desktop` file:

```bash
sudo cp target/release/exif-ai-gui /usr/local/bin/
sudo cp assets/icon_256.png /usr/share/icons/hicolor/256x256/apps/exif-ai.png
sudo cp assets/exif-ai-gui.desktop /usr/share/applications/
sudo update-desktop-database
```

### Cross-Compilation Targets

```bash
# macOS Apple Silicon
cargo build --release --target aarch64-apple-darwin

# macOS Intel
cargo build --release --target x86_64-apple-darwin

# Windows Intel
cargo build --release --target x86_64-pc-windows-msvc

# Windows ARM
cargo build --release --target aarch64-pc-windows-msvc
```

## Desktop GUI

The GUI provides a native desktop interface for processing images:

- **Drag & drop** images or use Open Files / Open Folder
- **Image preview** — thumbnails for JPEG, PNG, WebP, TIFF, HEIC, RAW (via macOS `sips` fallback)
- **Existing EXIF viewer** — grouped by Camera, Exposure, Image, Metadata, and GPS sections
- **AI processing** with real-time status and service failover
- **Results display** — title, description, tags, GPS, subject with write status
- **Dry Run mode** — preview AI results without modifying files
- **Settings panel** — configure AI services, EXIF fields, and output options
- **Model download** — download the local BLIP model from the GUI
- **Batch processing** — process multiple images at once

Run with:

```bash
cargo run --features gui --bin exif-ai-gui
```

Or launch the `.app` bundle from Finder on macOS.

## Quick Start (CLI)

```bash
# 1. Generate a default config file
exif-ai-cli --init

# 2. Edit config.json and add your API key(s)

# 3. Process a single image (dry run first)
exif-ai-cli --dry-run photo.jpg

# 4. Process for real
exif-ai-cli photo.jpg

# 5. Process an entire folder
exif-ai-cli ./photos/

# 6. Process multiple files with JSON output
exif-ai-cli --json photo1.jpg photo2.jpg

# 7. (Optional) Use local AI — no API keys needed
exif-ai-cli --download-model          # one-time ~1.75 GB download
# Then set "local.enabled": true in config.json
exif-ai-cli photo.jpg                  # uses local BLIP model
```

## Library Usage

### High-Level (Pipeline)

The simplest way to use the library — handles the full read → AI → write flow:

```rust
use exif_ai::config::Config;
use exif_ai::pipeline::{Pipeline, collect_images};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load config (contains API keys, field settings, etc.)
    let config = Config::load(Some("config.json".as_ref()))?;

    // Build the pipeline (services, fields, options — all from config)
    let pipeline = Pipeline::builder()
        .from_config(&config)
        .build()?;

    // Collect supported images from paths (files or directories, recursive)
    let images = collect_images(&[PathBuf::from("./photos")]);

    for path in &images {
        let result = pipeline.process_image(path).await;

        if let Some(ref err) = result.error {
            eprintln!("Error: {err}");
        } else {
            println!("Processed: {}", path.display());
            println!("  AI service: {:?}", result.ai_service_used);
            println!("  Title written: {}", result.title_written);

            // For HEIC/RAW: a sidecar .xmp file is written instead
            if let Some(ref sidecar) = result.sidecar_path {
                println!("  Sidecar XMP: {}", sidecar.display());
            }
        }
    }

    Ok(())
}
```

### Low-Level (Read / AI / Write separately)

For more control, call each step individually:

```rust
use exif_ai::exif::{read_exif, write_exif};
use exif_ai::ai::{OpenAiService, AiService, build_prompt};
use exif_ai::config::ExifFields;
use exif_ai::pipeline::ImageKind;
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = Path::new("photo.jpg");

    // 1. Read existing EXIF metadata
    let existing = read_exif(path)?;
    println!("Camera: {:?} {:?}", existing.make, existing.model);

    // 2. Analyze with AI
    let service = OpenAiService::new("sk-...".into(), "gpt-4o-mini".into());
    let bytes = std::fs::read(path)?;
    let b64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD, &bytes,
    );
    let ai_result = service.analyze(&b64, &build_prompt(), "image/jpeg").await?;
    println!("AI title: {:?}", ai_result.title);
    println!("AI tags: {:?}", ai_result.tags);

    // 3. Write metadata back (format-aware routing)
    let fields = ExifFields {
        write_title: true,
        write_description: true,
        write_tags: true,
        write_gps: true,
        write_subject: true,
        overwrite_existing: false,
    };
    let result = write_exif(path, &ai_result, &existing, &fields, false, ImageKind::Jpeg)?;
    println!("Title written: {}", result.title_written);

    Ok(())
}
```

### Key Types

| Type | Module | Purpose |
|------|--------|---------|
| [`Config`](config::Config) | `config` | All settings (API keys, fields, output) |
| [`ExifFields`](config::ExifFields) | `config` | Which fields to write + overwrite behavior |
| [`Pipeline`](pipeline::Pipeline) | `pipeline` | **Main entry point** — owns services + config, runs read → AI → write |
| [`PipelineBuilder`](pipeline::PipelineBuilder) | `pipeline` | Fluent builder for constructing a `Pipeline` |
| [`collect_images`](pipeline::collect_images) | `pipeline` | Walk paths, filter by supported extensions |
| [`ProcessResult`](pipeline::ProcessResult) | `pipeline` | What was written, errors, sidecar path |
| [`ImageKind`](pipeline::ImageKind) | `pipeline` | Format detection (Jpeg, Png, WebP, Tiff, Sidecar) |
| [`AiResult`](ai::AiResult) | `ai` | AI output (title, description, tags, gps, subject) |
| [`AiService`](ai::AiService) | `ai` | Trait for AI backends (implement for custom services) |
| [`ExifData`](exif::ExifData) | `exif` | Existing metadata read from a file |
| [`read_exif`](exif::read_exif) | `exif` | Read EXIF from any supported format |
| [`write_exif`](exif::write_exif) | `exif` | Write metadata (format-aware routing) |
| [`WriteResult`](exif::WriteResult) | `exif` | Which fields were written + sidecar path |

## Configuration

Run `exif-ai-cli --init` to generate a default `config.json` in the same directory as the binary.

```json
{
  "ai_services": {
    "local": {
      "model_path": "./models",
      "enabled": true
    },
    "openai": {
      "api_key": "sk-...",
      "model": "gpt-4o-mini",
      "enabled": true
    },
    "gemini": {
      "api_key": "AI...",
      "model": "gemini-2.0-flash",
      "enabled": false
    },
    "cloudflare": {
      "account_id": "",
      "api_token": "",
      "model": "@cf/llava-hf/llava-1.5-7b-hf",
      "enabled": false
    }
  },
  "service_order": ["local", "openai", "gemini", "cloudflare"],
  "exif_fields": {
    "write_title": true,
    "write_description": true,
    "write_tags": true,
    "write_gps": true,
    "write_subject": true,
    "overwrite_existing": false
  },
  "output": {
    "dry_run": false,
    "backup_originals": true,
    "log_file": null
  }
}
```

### AI Services

Configure one or more AI services. The `service_order` array determines the failover chain — if the first service fails or returns empty results, the next one is tried.

The local service is first in the default chain but disabled by default. If the model is missing when enabled, a warning is logged and the next service in the chain is tried.

| Service | Pricing | Notes |
|---------|---------|-------|
| **OpenAI** (GPT-4o-mini) | ~$0.001/image | Highest quality results |
| **Google Gemini** | Free tier: 15 req/min, 1,500/day | Great balance of quality and cost |
| **Cloudflare Workers AI** (LLaVA) | Free tier: ~100-200 images/day | Free but lower quality |
| **Local BLIP** (on-device) | Free forever | ~5s/image on CPU, no network needed |

### Metadata Written (Cross-Platform)

AI-generated metadata is written to **three industry standards** simultaneously for maximum compatibility across all platforms and tools:

#### EXIF (APP1 — TIFF/IFD)

| AI Output | EXIF Tags | IFD |
|-----------|-----------|-----|
| Title | `ImageDescription` (0x010E), `XPTitle` | IFD0 |
| Description | `UserComment` (0x9286), `XPComment` | ExifIFD / IFD0 |
| Tags | `XPKeywords` | IFD0 |
| Subject | `XPSubject` | IFD0 |
| GPS | `GPSLatitude`, `GPSLongitude` + refs | GPSIFD |

#### XMP (APP1 — XML)

| AI Output | XMP Property | Notes |
|-----------|-------------|-------|
| Title | `dc:title`, `photoshop:Headline` | Read by macOS, Linux, Adobe tools |
| Description | `dc:description` | Read by macOS Finder, Spotlight |
| Tags | `dc:subject` | Read by macOS, Lightroom, digiKam |

#### IPTC-IIM (APP13 — Photoshop 3.0)

| AI Output | IPTC Record | Notes |
|-----------|-------------|-------|
| Title | Object Name (2:5) | Read by older tools, Windows |
| Description | Caption/Abstract (2:120) | Broad compatibility |
| Tags | Keywords (2:25) | One record per keyword |

#### Platform Compatibility

| Platform | What's read |
|----------|-------------|
| **macOS** (Finder, Preview, Spotlight) | XMP, IPTC, EXIF ImageDescription |
| **Windows** (Explorer, Properties) | EXIF XP* tags, IPTC |
| **Linux** (digiKam, Shotwell, GNOME) | XMP, IPTC |
| **Adobe** (Lightroom, Photoshop, Bridge) | XMP, IPTC, EXIF |
| **exiftool** | All three standards |

> **Note:** GPS coordinates are only written when the image has no existing GPS data AND the AI identifies a known, real-world location.

## CLI Reference

```
Usage: exif-ai-cli [OPTIONS] [PATH]...

Arguments:
  [PATH]...  Image files or directories to process

Options:
  -c, --config <FILE>  Path to config file (default: config.json next to binary)
      --init           Initialize a default config.json and exit
      --dry-run        Preview changes without writing to files
      --json           Output results as JSON
  -v, --verbose        Verbose output
      --show-exif      Display all EXIF metadata and exit
      --clear-exif     Clear all EXIF/XMP/IPTC metadata from the image(s)
      --download-model Download the local BLIP model for offline inference
  -h, --help           Print help
  -V, --version        Print version
```

### Inspect EXIF

```bash
# Show all EXIF metadata for an image
exif-ai-cli --show-exif photo.jpg

# Show EXIF for multiple files
exif-ai-cli --show-exif photo1.jpg photo2.jpg ./photos/
```

### Clear EXIF

```bash
# Strip all EXIF/XMP/IPTC metadata from an image
exif-ai-cli --clear-exif photo.jpg

# Clear metadata from all images in a directory
exif-ai-cli --clear-exif ./photos/
```

> **Note:** Clearing EXIF from TIFF files is not supported (EXIF is integral to the TIFF structure). For HEIC/RAW sidecar formats, `--clear-exif` removes the `.xmp` sidecar file.

## How It Works

1. **Read** — Existing EXIF data is read using `nom-exif` (supports big-endian iPhone JPEGs, HEIC, RAW)
2. **Analyze** — The image is sent to the configured AI vision model for analysis
3. **Write** — AI-generated metadata is surgically injected into the file:
   - Original EXIF data is fully preserved (camera info, GPS, lens data, timestamps, etc.)
   - New tags are written to EXIF, XMP, and IPTC simultaneously
   - A `.bak` backup is created before any modification
4. **Verify** — Use `--dry-run` to preview what would be written without modifying files

## Supported Image Formats

| Format | Extensions | Read EXIF | Write Metadata | Strategy |
|--------|-----------|-----------|---------------|----------|
| **JPEG** | `.jpg`, `.jpeg` | ✅ | EXIF + XMP + IPTC | Native (in-place) |
| **PNG** | `.png` | ✅ | XMP (iTXt chunk) | Native (in-place) |
| **WebP** | `.webp` | ✅ | EXIF + XMP (RIFF) | Native (in-place) |
| **TIFF** | `.tif`, `.tiff` | ✅ | EXIF | Native (in-place) |
| **HEIC/HEIF** | `.heic`, `.heif` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **AVIF** | `.avif` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Canon RAW** | `.cr2`, `.cr3` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Adobe DNG** | `.dng` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Nikon RAW** | `.nef` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Sony RAW** | `.arw` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Fujifilm RAW** | `.raf` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Olympus RAW** | `.orf` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Panasonic RAW** | `.rw2` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Pentax RAW** | `.pef` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |
| **Samsung RAW** | `.srw` | ✅ | XMP sidecar (`.xmp`) | Sidecar file |

> **Sidecar files:** For HEIC, AVIF, and RAW formats, a `.xmp` sidecar file is written alongside the original. This is the industry-standard approach used by Lightroom, darktable, and digiKam — the original file is never modified.

## Requirements

- At least one AI service API key configured
- Rust 1.85+ (for building from source)

## License

MIT — see [LICENSE](LICENSE) for details.

## Author

**Sriram Govindan** — [GitHub](https://github.com/marirs/exif-ai)
