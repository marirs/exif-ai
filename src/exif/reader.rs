use anyhow::{Context, Result};
use nom_exif::*;
use std::path::Path;

// XP* tag IDs (IFD0)
const TAG_XP_TITLE: u16 = 0x9C9B;
const TAG_XP_COMMENT: u16 = 0x9C9C;
const TAG_XP_KEYWORDS: u16 = 0x9C9E;
const TAG_XP_SUBJECT: u16 = 0x9C9F;

/// Existing EXIF data extracted from an image.
#[derive(Debug, Clone, Default)]
pub struct ExifData {
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<String>,
    pub subject: Option<String>,
    pub has_gps: bool,
    pub gps_latitude: Option<f64>,
    pub gps_longitude: Option<f64>,

    // Standard EXIF fields for display
    pub make: Option<String>,
    pub model: Option<String>,
    pub date_time: Option<String>,
    pub orientation: Option<String>,
    pub x_resolution: Option<String>,
    pub y_resolution: Option<String>,
    pub software: Option<String>,
    pub exposure_time: Option<String>,
    pub f_number: Option<String>,
    pub iso: Option<String>,
    pub focal_length: Option<String>,
    pub color_space: Option<String>,
    pub image_width: Option<String>,
    pub image_height: Option<String>,
    pub lens_model: Option<String>,
}

/// Read existing EXIF data from an image file.
pub fn read_exif(path: &Path) -> Result<ExifData> {
    let mut parser = MediaParser::new();
    let ms = MediaSource::file_path(path).context("Failed to open image file")?;

    let iter: ExifIter = match parser.parse(ms) {
        Ok(iter) => iter,
        Err(_) => {
            log::debug!("No EXIF data found in {}", path.display());
            return Ok(ExifData::default());
        }
    };

    // Parse GPS info before converting to Exif (consumes the iterator)
    let gps_info = iter.parse_gps_info().ok().flatten();
    let exif: Exif = iter.into();

    let mut data = ExifData::default();

    // Title / ImageDescription
    if let Some(val) = exif.get(ExifTag::ImageDescription) {
        data.title = entry_to_string(val);
    }

    // XPTitle fallback
    if data.title.is_none() {
        if let Some(val) = exif.get_by_ifd_tag_code(0, TAG_XP_TITLE) {
            data.title = decode_xp_string(val).or_else(|| entry_to_string(val));
        }
    }

    // Description / UserComment
    if let Some(val) = exif.get(ExifTag::UserComment) {
        data.description = decode_user_comment(val).or_else(|| entry_to_string(val));
    }

    // XPComment fallback
    if data.description.is_none() {
        if let Some(val) = exif.get_by_ifd_tag_code(0, TAG_XP_COMMENT) {
            data.description = decode_xp_string(val).or_else(|| entry_to_string(val));
        }
    }

    // XPKeywords
    if let Some(val) = exif.get_by_ifd_tag_code(0, TAG_XP_KEYWORDS) {
        data.keywords = decode_xp_string(val).or_else(|| entry_to_string(val));
    }

    // XPSubject
    if let Some(val) = exif.get_by_ifd_tag_code(0, TAG_XP_SUBJECT) {
        data.subject = decode_xp_string(val).or_else(|| entry_to_string(val));
    }

    // Standard EXIF fields for display
    data.make = exif.get(ExifTag::Make).and_then(entry_to_string);
    data.model = exif.get(ExifTag::Model).and_then(entry_to_string);
    data.date_time = exif.get(ExifTag::DateTimeOriginal)
        .or_else(|| exif.get(ExifTag::CreateDate))
        .or_else(|| exif.get(ExifTag::ModifyDate))
        .and_then(entry_to_string);
    data.orientation = exif.get(ExifTag::Orientation).and_then(entry_to_string);
    data.software = exif.get(ExifTag::Software).and_then(entry_to_string);
    data.exposure_time = exif.get(ExifTag::ExposureTime).and_then(format_rational_frac);
    data.f_number = exif.get(ExifTag::FNumber).and_then(|v| {
        format_rational_decimal(v).map(|s| format!("f/{s}"))
    });
    data.iso = exif.get(ExifTag::ISOSpeedRatings).and_then(entry_to_string);
    data.focal_length = exif.get(ExifTag::FocalLength).and_then(|v| {
        format_rational_decimal(v).map(|s| format!("{s} mm"))
    });
    data.color_space = exif.get(ExifTag::ColorSpace).and_then(|v| {
        v.as_u16().map(|c| match c {
            1 => "sRGB".to_string(),
            65535 => "Uncalibrated".to_string(),
            _ => format!("{c}"),
        })
    });
    data.image_width = exif.get(ExifTag::ExifImageWidth)
        .or_else(|| exif.get(ExifTag::ImageWidth))
        .and_then(entry_to_string);
    data.image_height = exif.get(ExifTag::ExifImageHeight)
        .or_else(|| exif.get(ExifTag::ImageHeight))
        .and_then(entry_to_string);
    data.lens_model = exif.get(ExifTag::LensModel).and_then(entry_to_string);
    data.x_resolution = exif.get(ExifTag::XResolution).and_then(|v| {
        format_rational_decimal(v).map(|s| format!("{s} dpi"))
    });
    data.y_resolution = exif.get(ExifTag::YResolution).and_then(|v| {
        format_rational_decimal(v).map(|s| format!("{s} dpi"))
    });

    // GPS — use nom-exif's built-in GPS parser
    if let Some(gps) = gps_info {
        data.has_gps = true;
        data.gps_latitude = Some(latlng_to_decimal(&gps.latitude, gps.latitude_ref));
        data.gps_longitude = Some(latlng_to_decimal(&gps.longitude, gps.longitude_ref));
    }

    Ok(data)
}

/// Convert an EntryValue to an Option<String>.
fn entry_to_string(val: &EntryValue) -> Option<String> {
    let s = val.to_string();
    let s = s.trim().trim_matches('"').to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// Decode a UserComment EntryValue (UNDEFINED format with 8-byte charset prefix).
fn decode_user_comment(val: &EntryValue) -> Option<String> {
    if let EntryValue::Undefined(bytes) = val {
        if bytes.len() > 8 {
            let prefix = &bytes[0..8];
            let payload = &bytes[8..];
            if prefix == b"ASCII\0\0\0" {
                let s = String::from_utf8_lossy(payload).trim().to_string();
                if !s.is_empty() { return Some(s); }
            } else if prefix == b"UNICODE\0" {
                // UTF-16 encoded
                return decode_utf16le(payload);
            }
        }
    }
    None
}

/// Decode a UTF-16LE byte array (XP* tags) into a String.
fn decode_xp_string(val: &EntryValue) -> Option<String> {
    let bytes = match val {
        EntryValue::U8Array(b) => b.as_slice(),
        EntryValue::Undefined(b) => b.as_slice(),
        _ => return None,
    };
    decode_utf16le(bytes)
}

/// Decode raw UTF-16LE bytes into a String.
fn decode_utf16le(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 2 { return None; }
    let u16s: Vec<u16> = bytes.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let s = String::from_utf16_lossy(&u16s)
        .trim_end_matches('\0')
        .trim()
        .to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// Format a rational EntryValue as a fraction string (e.g. "1/4310").
fn format_rational_frac(val: &EntryValue) -> Option<String> {
    if let Some(r) = val.as_urational() {
        if r.1 == 0 { return None; }
        if r.0 == 0 { return Some("0".to_string()); }
        // If denominator is 1, just show the number
        if r.1 == 1 {
            return Some(format!("{}", r.0));
        }
        Some(format!("{}/{}", r.0, r.1))
    } else {
        entry_to_string(val)
    }
}

/// Format a rational EntryValue as a clean decimal string (e.g. "1.78").
fn format_rational_decimal(val: &EntryValue) -> Option<String> {
    if let Some(r) = val.as_urational() {
        if r.1 == 0 { return None; }
        let decimal = r.0 as f64 / r.1 as f64;
        // Remove unnecessary trailing zeros
        let formatted = format!("{:.2}", decimal);
        let formatted = formatted.trim_end_matches('0').trim_end_matches('.').to_string();
        Some(formatted)
    } else {
        entry_to_string(val)
    }
}

/// Convert a nom-exif LatLng (3 URationals: deg, min, sec) to decimal degrees.
fn latlng_to_decimal(latlng: &LatLng, reference: char) -> f64 {
    let degrees = latlng.0.0 as f64 / latlng.0.1 as f64;
    let minutes = latlng.1.0 as f64 / latlng.1.1 as f64;
    let seconds = latlng.2.0 as f64 / latlng.2.1 as f64;

    let mut coord = degrees + minutes / 60.0 + seconds / 3600.0;

    if reference == 'S' || reference == 'W' {
        coord = -coord;
    }

    coord
}
