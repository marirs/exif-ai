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
            data.title = entry_to_string(val);
        }
    }

    // Description / UserComment
    if let Some(val) = exif.get(ExifTag::UserComment) {
        data.description = entry_to_string(val);
    }

    // XPComment fallback
    if data.description.is_none() {
        if let Some(val) = exif.get_by_ifd_tag_code(0, TAG_XP_COMMENT) {
            data.description = entry_to_string(val);
        }
    }

    // XPKeywords
    if let Some(val) = exif.get_by_ifd_tag_code(0, TAG_XP_KEYWORDS) {
        data.keywords = entry_to_string(val);
    }

    // XPSubject
    if let Some(val) = exif.get_by_ifd_tag_code(0, TAG_XP_SUBJECT) {
        data.subject = entry_to_string(val);
    }

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
