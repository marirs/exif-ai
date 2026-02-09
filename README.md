# exif-ai

AI-powered EXIF metadata writer — generate SEO titles, descriptions, tags, GPS, and subject data for images using AI vision models.

## Features

- **AI Vision Analysis** — Send images to AI models for intelligent metadata generation
- **Multi-Service Failover** — Configurable chain: OpenAI GPT-4o-mini → Google Gemini → Cloudflare Workers AI
- **EXIF Writing** — Writes title, description, tags, GPS coordinates, and subject identification directly into image EXIF data
- **GPS Intelligence** — Only writes GPS coordinates when the image has no existing GPS data AND the AI identifies a known location
- **Subject Detection** — Identifies known people, bird species, animal species, and landmarks
- **Batch Processing** — Process single files, multiple files, or entire directories recursively
- **Backup Originals** — Automatically backs up images before modifying EXIF data
- **Dry Run Mode** — Preview what would be written without modifying any files
- **JSON Output** — Machine-readable output for scripting and automation
- **Cross-Platform** — Builds for macOS (Silicon & Intel) and Windows (ARM & Intel)

## Installation

### From Source

```bash
git clone https://github.com/marirs/exif-ai.git
cd exif-ai
cargo build --release
```

The binary will be at `target/release/exif-ai`.

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
exif-ai --init

# 2. Edit config.json and add your API key(s)

# 3. Process a single image (dry run first)
exif-ai --dry-run photo.jpg

# 4. Process for real
exif-ai photo.jpg

# 5. Process an entire folder
exif-ai ./photos/

# 6. Process multiple files with JSON output
exif-ai --json photo1.jpg photo2.jpg
```

## Configuration

Run `exif-ai --init` to generate a default `config.json` in the same directory as the binary.

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

### EXIF Fields Written

| AI Output | EXIF Tags | Notes |
|-----------|-----------|-------|
| Title | `ImageDescription`, `XPTitle` | SEO title, max 60 chars |
| Description | `UserComment`, `XPComment` | SEO description, max 254 chars |
| Tags | `XPKeywords` | Semicolon-separated keywords |
| Subject | `XPSubject` | Identified people, animals, landmarks |
| GPS | `GPSLatitude`, `GPSLongitude` + refs | Only if no existing GPS AND location identified |

## CLI Reference

```
Usage: exif-ai [OPTIONS] [PATH]...

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

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- TIFF (`.tif`, `.tiff`)

## Requirements

- At least one AI service API key configured
- Rust 1.85+ (for building from source)

## License

MIT — see [LICENSE](LICENSE) for details.

## Author

**Sriram Govindan** — [GitHub](https://github.com/marirs/exif-ai)
