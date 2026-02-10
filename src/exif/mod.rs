//! EXIF, XMP, and IPTC metadata reading and writing.
//!
//! This module provides two main functions:
//!
//! - [`read_exif`] — Read existing metadata from any supported image format
//! - [`write_exif`] — Write AI-generated metadata back to the image (format-aware)
//!
//! The writer automatically routes to the correct strategy based on [`ImageKind`](crate::pipeline::ImageKind):
//! JPEG gets EXIF+XMP+IPTC, PNG gets XMP, WebP gets EXIF+XMP, TIFF gets EXIF,
//! and HEIC/RAW formats get a sidecar `.xmp` file.

mod reader;
mod writer;

pub use reader::{ExifData, read_exif};
pub use writer::{clear_exif, write_exif, WriteResult};
