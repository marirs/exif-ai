# exif-ai

AI-powered EXIF metadata writer — generate SEO titles, descriptions, tags, GPS, and subject data for images using AI vision models.

## Features

- **AI Vision Analysis** — Send images to AI models for intelligent metadata generation
- **Multi-Service Failover** — Configurable chain: OpenAI GPT-4o-mini → Google Gemini → Cloudflare Workers AI
- **Multi-Format Support** — JPEG, PNG, WebP, TIFF (native write), HEIC/HEIF, AVIF, and 10+ RAW formats (sidecar XMP)
- **EXIF Writing** — Writes title, description, tags, GPS coordinates, and subject identification directly into image EXIF data
- **GPS Intelligence** — Only writes GPS coordinates when the image has no existing GPS data AND the AI identifies a known location
- **Subject Detection** — Identifies known people, bird species, animal species, and landmarks
- **Batch Processing** — Process single files, multiple files, or entire directories recursively
- **Backup Originals** — Automatically backs up images before modifying EXIF data
- **Dry Run Mode** — Preview what would be written without modifying any files
- **JSON Output** — Machine-readable output for scripting and automation
- **Cross-Platform** — Builds for macOS (Silicon & Intel) and Windows (ARM & Intel)

## Installation

### As a Library

```toml
[dependencies]
exif-ai = { version = "0.1", default-features = false }
```

### CLI Binary (from source)

```bash
git clone https://github.com/marirs/exif-ai.git
cd exif-ai
cargo build --release
```

The binary will be at `target/release/exif-ai-cli`.

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

## Quick Start

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
```

## Configuration

Run `exif-ai-cli --init` to generate a default `config.json` in the same directory as the binary.

```json
{
  "ai_services": {
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
  "service_order": ["openai", "gemini", "cloudflare"],
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

| Service | Pricing | Notes |
|---------|---------|-------|
| **OpenAI** (GPT-4o-mini) | ~$0.001/image | Highest quality results |
| **Google Gemini** | Free tier: 15 req/min, 1,500/day | Great balance of quality and cost |
| **Cloudflare Workers AI** (LLaVA) | Free tier: ~100-200 images/day | Free but lower quality |

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
  -h, --help           Print help
  -V, --version        Print version
```

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
