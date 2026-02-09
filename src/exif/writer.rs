use anyhow::{Context, Result};
use little_exif::endian::Endian;
use little_exif::exif_tag::{ExifTag, ExifTagGroup};
use little_exif::exif_tag_format::ExifTagFormat;
use little_exif::metadata::Metadata;
use std::path::Path;

use crate::ai::{AiResult, GpsCoords};
use crate::config::ExifFields;
use super::reader::ExifData;

// EXIF tag IDs for tags not natively supported by little_exif
const TAG_XP_TITLE: u16 = 0x9C9B;
const TAG_XP_COMMENT: u16 = 0x9C9C;
const TAG_XP_KEYWORDS: u16 = 0x9C9E;
const TAG_XP_SUBJECT: u16 = 0x9C9F;
const TAG_GPS_LATITUDE_REF: u16 = 0x0001;
const TAG_GPS_LATITUDE: u16 = 0x0002;
const TAG_GPS_LONGITUDE_REF: u16 = 0x0003;
const TAG_GPS_LONGITUDE: u16 = 0x0004;

/// Result of writing EXIF data to an image.
#[derive(Debug, Default)]
pub struct WriteResult {
    pub title_written: bool,
    pub description_written: bool,
    pub tags_written: bool,
    pub gps_written: bool,
    pub subject_written: bool,
    pub skipped_fields: Vec<String>,
}

/// Encode a string as UTF-16LE bytes (used for XP* tags).
fn encode_utf16le(s: &str) -> Vec<u8> {
    let mut bytes: Vec<u8> = s
        .encode_utf16()
        .flat_map(|c| c.to_le_bytes())
        .collect();
    // Null terminator
    bytes.push(0);
    bytes.push(0);
    bytes
}

/// Create an XP* tag (UTF-16LE encoded, IFD0 group).
fn make_xp_tag(tag_id: u16, value: &str) -> Option<ExifTag> {
    let raw_data = encode_utf16le(value);
    ExifTag::from_u16_with_data(
        tag_id,
        &ExifTagFormat::INT8U,
        &raw_data,
        &Endian::Little,
        &ExifTagGroup::IFD0,
    )
    .ok()
}

/// Encode a GPS rational value as raw bytes (3 rationals = 24 bytes, little-endian).
fn encode_gps_rational(degrees: u32, minutes: u32, seconds_num: u32, seconds_den: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(24);
    bytes.extend_from_slice(&degrees.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&minutes.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&seconds_num.to_le_bytes());
    bytes.extend_from_slice(&seconds_den.to_le_bytes());
    bytes
}

/// Write AI-generated metadata into the image EXIF.
pub fn write_exif(
    path: &Path,
    ai_result: &AiResult,
    existing: &ExifData,
    fields: &ExifFields,
    dry_run: bool,
) -> Result<WriteResult> {
    let mut result = WriteResult::default();

    let mut metadata = match Metadata::new_from_path(path) {
        Ok(m) => m,
        Err(_) => Metadata::new(),
    };

    // Title — ImageDescription (native) + XPTitle (custom)
    if fields.write_title {
        if let Some(ref title) = ai_result.title {
            if existing.title.is_none() || fields.overwrite_existing {
                metadata.set_tag(ExifTag::ImageDescription(title.clone()));
                if let Some(xp_tag) = make_xp_tag(TAG_XP_TITLE, title) {
                    metadata.set_tag(xp_tag);
                }
                result.title_written = true;
                log::debug!("  Title: {title}");
            } else {
                result.skipped_fields.push("title (existing)".to_string());
            }
        }
    }

    // Description — UserComment (native) + XPComment (custom)
    if fields.write_description {
        if let Some(ref desc) = ai_result.description {
            if existing.description.is_none() || fields.overwrite_existing {
                // UserComment requires ASCII prefix (8 bytes) + content bytes
                let mut comment_bytes = b"ASCII\0\0\0".to_vec();
                comment_bytes.extend_from_slice(desc.as_bytes());
                metadata.set_tag(ExifTag::UserComment(comment_bytes));
                if let Some(xp_tag) = make_xp_tag(TAG_XP_COMMENT, desc) {
                    metadata.set_tag(xp_tag);
                }
                result.description_written = true;
                log::debug!("  Description: {desc}");
            } else {
                result
                    .skipped_fields
                    .push("description (existing)".to_string());
            }
        }
    }

    // Tags / Keywords — XPKeywords (custom)
    if fields.write_tags {
        if let Some(ref tags) = ai_result.tags {
            if existing.keywords.is_none() || fields.overwrite_existing {
                let keywords_str = tags.join("; ");
                if let Some(xp_tag) = make_xp_tag(TAG_XP_KEYWORDS, &keywords_str) {
                    metadata.set_tag(xp_tag);
                }
                result.tags_written = true;
                log::debug!("  Tags: {}", tags.join(", "));
            } else {
                result.skipped_fields.push("tags (existing)".to_string());
            }
        }
    }

    // Subject — XPSubject (custom)
    if fields.write_subject {
        if let Some(ref subjects) = ai_result.subject {
            if !subjects.is_empty() && (existing.subject.is_none() || fields.overwrite_existing) {
                let subject_str = subjects.join("; ");
                if let Some(xp_tag) = make_xp_tag(TAG_XP_SUBJECT, &subject_str) {
                    metadata.set_tag(xp_tag);
                }
                result.subject_written = true;
                log::debug!("  Subject: {}", subjects.join(", "));
            } else if !subjects.is_empty() {
                result
                    .skipped_fields
                    .push("subject (existing)".to_string());
            }
        }
    }

    // GPS — only if no existing GPS AND AI identified a location
    if fields.write_gps {
        if let Some(ref gps) = ai_result.gps {
            if !existing.has_gps {
                write_gps_tags(&mut metadata, gps);
                result.gps_written = true;
                log::debug!("  GPS: {}, {}", gps.latitude, gps.longitude);
            } else {
                result
                    .skipped_fields
                    .push("gps (existing coordinates)".to_string());
            }
        }
    }

    // Write to file
    if !dry_run {
        metadata
            .write_to_file(path)
            .context("Failed to write EXIF metadata to file")?;
    }

    Ok(result)
}

/// Write GPS latitude and longitude EXIF tags using raw byte construction.
fn write_gps_tags(metadata: &mut Metadata, gps: &GpsCoords) {
    let lat = gps.latitude;
    let lon = gps.longitude;

    let lat_ref = if lat >= 0.0 { "N" } else { "S" };
    let lon_ref = if lon >= 0.0 { "E" } else { "W" };

    let lat_abs = lat.abs();
    let lon_abs = lon.abs();

    let lat_deg = lat_abs.floor() as u32;
    let lat_min = ((lat_abs - lat_deg as f64) * 60.0).floor() as u32;
    let lat_sec = ((lat_abs - lat_deg as f64 - lat_min as f64 / 60.0) * 3600.0 * 10000.0) as u32;

    let lon_deg = lon_abs.floor() as u32;
    let lon_min = ((lon_abs - lon_deg as f64) * 60.0).floor() as u32;
    let lon_sec = ((lon_abs - lon_deg as f64 - lon_min as f64 / 60.0) * 3600.0 * 10000.0) as u32;

    // GPSLatitudeRef
    if let Ok(tag) = ExifTag::from_u16_with_data(
        TAG_GPS_LATITUDE_REF,
        &ExifTagFormat::STRING,
        &format!("{lat_ref}\0").into_bytes(),
        &Endian::Little,
        &ExifTagGroup::GPSIFD,
    ) {
        metadata.set_tag(tag);
    }

    // GPSLatitude (3 rationals)
    let lat_bytes = encode_gps_rational(lat_deg, lat_min, lat_sec, 10000);
    if let Ok(tag) = ExifTag::from_u16_with_data(
        TAG_GPS_LATITUDE,
        &ExifTagFormat::RATIONAL64U,
        &lat_bytes,
        &Endian::Little,
        &ExifTagGroup::GPSIFD,
    ) {
        metadata.set_tag(tag);
    }

    // GPSLongitudeRef
    if let Ok(tag) = ExifTag::from_u16_with_data(
        TAG_GPS_LONGITUDE_REF,
        &ExifTagFormat::STRING,
        &format!("{lon_ref}\0").into_bytes(),
        &Endian::Little,
        &ExifTagGroup::GPSIFD,
    ) {
        metadata.set_tag(tag);
    }

    // GPSLongitude (3 rationals)
    let lon_bytes = encode_gps_rational(lon_deg, lon_min, lon_sec, 10000);
    if let Ok(tag) = ExifTag::from_u16_with_data(
        TAG_GPS_LONGITUDE,
        &ExifTagFormat::RATIONAL64U,
        &lon_bytes,
        &Endian::Little,
        &ExifTagGroup::GPSIFD,
    ) {
        metadata.set_tag(tag);
    }
}
