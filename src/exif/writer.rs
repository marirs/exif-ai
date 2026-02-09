use anyhow::{Context, Result};
use img_parts::Bytes;
use img_parts::jpeg::{Jpeg, JpegSegment};
use img_parts::ImageEXIF;
use little_exif::endian::Endian;
use little_exif::exif_tag::{ExifTag, ExifTagGroup};
use little_exif::exif_tag_format::ExifTagFormat;
use little_exif::filetype::FileExtension;
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

// little_exif as_u8_vec(JPEG) returns: [APP1 marker 2B][length 2B][Exif\0\0 6B][TIFF data]
// img-parts set_exif() expects just the TIFF data (after Exif\0\0)
const JPEG_EXIF_OVERHEAD: usize = 10; // 2 + 2 + 6

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

/// Load existing EXIF metadata from a file path using little_exif.
/// Returns None if it can't parse (instead of losing data).
fn load_existing_metadata(path: &Path) -> Option<Metadata> {
    let path_owned = path.to_path_buf();
    // Suppress panics from little_exif
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(move || {
        Metadata::new_from_path(&path_owned)
    });
    std::panic::set_hook(prev_hook);

    match result {
        Ok(Ok(m)) => {
            // Check if little_exif actually loaded any tags
            if m.data().is_empty() {
                log::debug!("little_exif loaded empty metadata");
                None
            } else {
                log::debug!("little_exif loaded {} existing EXIF tags", m.data().len());
                Some(m)
            }
        }
        Ok(Err(e)) => {
            log::debug!("little_exif could not parse EXIF: {e}");
            None
        }
        Err(_) => {
            log::debug!("little_exif panicked parsing EXIF");
            None
        }
    }
}

/// Write AI-generated metadata into the image EXIF, preserving all existing data.
///
/// Strategy:
/// 1. Read the entire JPEG with img-parts (preserves all segments)
/// 2. Try to load existing EXIF with little_exif and merge AI tags
/// 3. If little_exif can't parse, inject AI tags into the raw EXIF via img-parts
/// 4. Write back via img-parts (only APP1 EXIF segment changes)
pub fn write_exif(
    path: &Path,
    ai_result: &AiResult,
    existing: &ExifData,
    fields: &ExifFields,
    dry_run: bool,
) -> Result<WriteResult> {
    let mut result = WriteResult::default();

    // Collect which tags to write (used for both dry-run and real write)
    let mut new_tags: Vec<ExifTag> = Vec::new();

    // Title — ImageDescription (native) + XPTitle (custom)
    if fields.write_title {
        if let Some(ref title) = ai_result.title {
            if existing.title.is_none() || fields.overwrite_existing {
                new_tags.push(ExifTag::ImageDescription(title.clone()));
                if let Some(xp_tag) = make_xp_tag(TAG_XP_TITLE, title) {
                    new_tags.push(xp_tag);
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
                let mut comment_bytes = b"ASCII\0\0\0".to_vec();
                comment_bytes.extend_from_slice(desc.as_bytes());
                new_tags.push(ExifTag::UserComment(comment_bytes));
                if let Some(xp_tag) = make_xp_tag(TAG_XP_COMMENT, desc) {
                    new_tags.push(xp_tag);
                }
                result.description_written = true;
                log::debug!("  Description: {desc}");
            } else {
                result.skipped_fields.push("description (existing)".to_string());
            }
        }
    }

    // Tags / Keywords — XPKeywords (custom)
    if fields.write_tags {
        if let Some(ref tags) = ai_result.tags {
            if existing.keywords.is_none() || fields.overwrite_existing {
                let keywords_str = tags.join("; ");
                if let Some(xp_tag) = make_xp_tag(TAG_XP_KEYWORDS, &keywords_str) {
                    new_tags.push(xp_tag);
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
                    new_tags.push(xp_tag);
                }
                result.subject_written = true;
                log::debug!("  Subject: {}", subjects.join(", "));
            } else if !subjects.is_empty() {
                result.skipped_fields.push("subject (existing)".to_string());
            }
        }
    }

    // GPS — only if no existing GPS AND AI identified a location
    if fields.write_gps {
        if let Some(ref gps) = ai_result.gps {
            if !existing.has_gps {
                collect_gps_tags(&mut new_tags, gps);
                result.gps_written = true;
                log::debug!("  GPS: {}, {}", gps.latitude, gps.longitude);
            } else {
                result.skipped_fields.push("gps (existing coordinates)".to_string());
            }
        }
    }

    // Write to file using img-parts to preserve existing JPEG structure
    if !dry_run && !new_tags.is_empty() {
        write_tags_to_jpeg(path, &new_tags, ai_result, existing, fields)
            .context("Failed to write EXIF metadata to file")?;
    }

    Ok(result)
}

/// Write new EXIF tags into a JPEG file, preserving all existing data.
fn write_tags_to_jpeg(
    path: &Path,
    new_tags: &[ExifTag],
    ai_result: &AiResult,
    existing: &ExifData,
    fields: &ExifFields,
) -> Result<()> {
    let file_bytes = std::fs::read(path).context("Failed to read image file")?;

    // Parse JPEG structure with img-parts (preserves all segments)
    let mut jpeg = Jpeg::from_bytes(Bytes::from(file_bytes.clone()))
        .map_err(|e| anyhow::anyhow!("Failed to parse JPEG: {e}"))?;

    // Remember where the EXIF segment was originally positioned
    let orig_exif_pos = find_exif_segment_pos(&jpeg);
    let original_exif = jpeg.exif().unwrap_or_default();

    // Build the new EXIF TIFF data
    let new_tiff_data: Option<Vec<u8>>;

    // Try the little_exif round-trip first (works when it can parse the EXIF)
    if let Some(mut metadata) = load_existing_metadata(path) {
        log::debug!("little_exif parsed existing EXIF, using merge strategy");
        for tag in new_tags {
            metadata.set_tag(tag.clone());
        }
        let exif_bytes = metadata.as_u8_vec(FileExtension::JPEG);
        if exif_bytes.len() > JPEG_EXIF_OVERHEAD {
            new_tiff_data = Some(exif_bytes[JPEG_EXIF_OVERHEAD..].to_vec());
        } else {
            new_tiff_data = None;
        }
    } else {
        log::info!("Using raw TIFF injection to preserve original EXIF");
        if original_exif.is_empty() {
            // No existing EXIF — build fresh
            let mut metadata = Metadata::new();
            for tag in new_tags {
                metadata.set_tag(tag.clone());
            }
            let exif_bytes = metadata.as_u8_vec(FileExtension::JPEG);
            if exif_bytes.len() > JPEG_EXIF_OVERHEAD {
                new_tiff_data = Some(exif_bytes[JPEG_EXIF_OVERHEAD..].to_vec());
            } else {
                new_tiff_data = None;
            }
        } else {
            let merged = inject_ai_tags_into_tiff(&original_exif, ai_result, existing, fields)?;
            new_tiff_data = Some(merged);
        }
    }

    // Write the new EXIF via set_exif (removes old, inserts at pos 3)
    if let Some(tiff_data) = new_tiff_data {
        jpeg.set_exif(Some(Bytes::from(tiff_data)));

        // set_exif() inserts at position 3, which may be after XMP APP1.
        // Move the EXIF segment back to its original position so EXIF comes
        // before XMP (required for many EXIF parsers).
        let new_exif_pos = find_exif_segment_pos(&jpeg);
        if let Some(new_pos) = new_exif_pos {
            let target_pos = orig_exif_pos.unwrap_or(1); // default: right after APP0
            if new_pos != target_pos && target_pos < new_pos {
                let segments = jpeg.segments_mut();
                let seg = segments.remove(new_pos);
                segments.insert(target_pos, seg);
            }
        }
    }

    // === Write XMP metadata (dc:title, dc:description, dc:subject) ===
    update_xmp_metadata(&mut jpeg, ai_result, existing, fields);

    // === Write IPTC metadata (caption, keywords) ===
    update_iptc_metadata(&mut jpeg, ai_result, existing, fields);

    let output = jpeg.encoder().bytes();
    std::fs::write(path, &output).context("Failed to write JPEG file")?;

    Ok(())
}

/// Find the position of the EXIF APP1 segment in a JPEG.
/// EXIF segments have marker 0xE1 (APP1) and contents starting with "Exif\0\0".
fn find_exif_segment_pos(jpeg: &Jpeg) -> Option<usize> {
    const EXIF_PREFIX: &[u8] = b"Exif\0\0";
    jpeg.segments().iter().position(|s| {
        s.marker() == 0xE1 && s.contents().starts_with(EXIF_PREFIX)
    })
}

// ============================================================================
// XMP Metadata Writing
// ============================================================================

const XMP_HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

/// Find the XMP APP1 segment position in a JPEG.
fn find_xmp_segment_pos(jpeg: &Jpeg) -> Option<usize> {
    jpeg.segments().iter().position(|s| {
        s.marker() == 0xE1 && s.contents().starts_with(XMP_HEADER)
    })
}

/// Update or create XMP metadata in the JPEG with AI-generated fields.
/// Writes dc:title, dc:description, dc:subject for macOS/Linux/Adobe compatibility.
fn update_xmp_metadata(
    jpeg: &mut Jpeg,
    ai_result: &AiResult,
    existing: &ExifData,
    fields: &ExifFields,
) {
    // Collect what we need to write
    let title = if fields.write_title {
        ai_result.title.as_ref()
            .filter(|_| existing.title.is_none() || fields.overwrite_existing)
    } else { None };

    let description = if fields.write_description {
        ai_result.description.as_ref()
            .filter(|_| existing.description.is_none() || fields.overwrite_existing)
    } else { None };

    let keywords: Option<&Vec<String>> = if fields.write_tags {
        ai_result.tags.as_ref()
            .filter(|_| existing.keywords.is_none() || fields.overwrite_existing)
    } else { None };

    if title.is_none() && description.is_none() && keywords.is_none() {
        return;
    }

    // Try to read existing XMP
    let xmp_pos = find_xmp_segment_pos(jpeg);
    let existing_xmp = xmp_pos.map(|pos| {
        let contents = jpeg.segments()[pos].contents();
        let xmp_bytes = &contents[XMP_HEADER.len()..];
        String::from_utf8_lossy(xmp_bytes).to_string()
    });

    // Build the new XMP
    let new_xmp = build_xmp(existing_xmp.as_deref(), title.map(|s| s.as_str()), description.map(|s| s.as_str()), keywords);

    // Build the segment contents: XMP header + XMP data
    let mut contents = Vec::with_capacity(XMP_HEADER.len() + new_xmp.len());
    contents.extend_from_slice(XMP_HEADER);
    contents.extend_from_slice(new_xmp.as_bytes());

    let new_segment = JpegSegment::new_with_contents(0xE1, Bytes::from(contents));

    let segments = jpeg.segments_mut();
    if let Some(pos) = xmp_pos {
        segments[pos] = new_segment;
    } else {
        // Insert after EXIF APP1 or at position 2
        let insert_pos = find_exif_segment_pos_from_segments(segments)
            .map(|p| p + 1)
            .unwrap_or(2);
        let insert_pos = std::cmp::min(insert_pos, segments.len());
        segments.insert(insert_pos, new_segment);
    }
}

/// Helper to find EXIF segment position from a segments slice.
fn find_exif_segment_pos_from_segments(segments: &[JpegSegment]) -> Option<usize> {
    const EXIF_PREFIX: &[u8] = b"Exif\0\0";
    segments.iter().position(|s| {
        s.marker() == 0xE1 && s.contents().starts_with(EXIF_PREFIX)
    })
}

/// Build XMP XML string, preserving existing XMP content and injecting new fields.
fn build_xmp(
    existing: Option<&str>,
    title: Option<&str>,
    description: Option<&str>,
    keywords: Option<&Vec<String>>,
) -> String {
    // If we have existing XMP, try to inject into it
    if let Some(xmp) = existing {
        return inject_into_existing_xmp(xmp, title, description, keywords);
    }

    // Build fresh XMP
    let mut xmp = String::new();
    xmp.push_str("<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n");
    xmp.push_str("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n");
    xmp.push_str("<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n");
    xmp.push_str("<rdf:Description rdf:about=\"\"\n");
    xmp.push_str("  xmlns:dc=\"http://purl.org/dc/elements/1.1/\"\n");
    xmp.push_str("  xmlns:photoshop=\"http://ns.adobe.com/photoshop/1.0/\">\n");

    if let Some(t) = title {
        let t_esc = xml_escape(t);
        xmp.push_str(&format!("  <dc:title><rdf:Alt><rdf:li xml:lang=\"x-default\">{t_esc}</rdf:li></rdf:Alt></dc:title>\n"));
        xmp.push_str(&format!("  <photoshop:Headline>{t_esc}</photoshop:Headline>\n"));
    }

    if let Some(d) = description {
        let d_esc = xml_escape(d);
        xmp.push_str(&format!("  <dc:description><rdf:Alt><rdf:li xml:lang=\"x-default\">{d_esc}</rdf:li></rdf:Alt></dc:description>\n"));
        xmp.push_str(&format!("  <photoshop:CaptionWriter>AI</photoshop:CaptionWriter>\n"));
    }

    if let Some(kw) = keywords {
        xmp.push_str("  <dc:subject><rdf:Bag>\n");
        for k in kw {
            xmp.push_str(&format!("    <rdf:li>{}</rdf:li>\n", xml_escape(k)));
        }
        xmp.push_str("  </rdf:Bag></dc:subject>\n");
    }

    xmp.push_str("</rdf:Description>\n");
    xmp.push_str("</rdf:RDF>\n");
    xmp.push_str("</x:xmpmeta>\n");
    xmp.push_str("<?xpacket end=\"w\"?>");
    xmp
}

/// Inject dc:title, dc:description, dc:subject into existing XMP XML.
fn inject_into_existing_xmp(
    xmp: &str,
    title: Option<&str>,
    description: Option<&str>,
    keywords: Option<&Vec<String>>,
) -> String {
    let mut result = xmp.to_string();

    // Ensure dc namespace is declared
    if !result.contains("xmlns:dc=") {
        if let Some(pos) = result.find("rdf:about=\"\"") {
            let insert_at = pos + "rdf:about=\"\"".len();
            result.insert_str(insert_at, "\n  xmlns:dc=\"http://purl.org/dc/elements/1.1/\"");
        }
    }

    // Ensure photoshop namespace is declared
    if !result.contains("xmlns:photoshop=") {
        if let Some(pos) = result.find("rdf:about=\"\"") {
            let insert_at = pos + "rdf:about=\"\"".len();
            result.insert_str(insert_at, "\n  xmlns:photoshop=\"http://ns.adobe.com/photoshop/1.0/\"");
        }
    }

    // Find insertion point: before </rdf:Description>
    let insert_before = result.find("</rdf:Description>")
        .or_else(|| result.find("/>").and_then(|p| {
            // Check if this is the self-closing rdf:Description
            if result[..p].rfind('<').map(|s| result[s..].starts_with("<rdf:Description")).unwrap_or(false) {
                // Convert self-closing to open/close
                None
            } else {
                None
            }
        }));

    // Handle self-closing rdf:Description: convert to open/close
    if insert_before.is_none() {
        // Find the self-closing rdf:Description
        if let Some(desc_start) = result.find("<rdf:Description") {
            if let Some(close_pos) = result[desc_start..].find("/>") {
                let abs_close = desc_start + close_pos;
                result.replace_range(abs_close..abs_close + 2, ">");
                // Find </rdf:RDF> and insert </rdf:Description> before it
                if let Some(rdf_end) = result.find("</rdf:RDF>") {
                    result.insert_str(rdf_end, "</rdf:Description>\n");
                }
            }
        }
    }

    // Now find the insertion point
    if result.find("</rdf:Description>").is_some() {
        let mut new_elements = String::new();

        if let Some(t) = title {
            let t_esc = xml_escape(t);
            // Remove existing dc:title if present
            remove_xml_element(&mut result, "dc:title");
            new_elements.push_str(&format!("  <dc:title><rdf:Alt><rdf:li xml:lang=\"x-default\">{t_esc}</rdf:li></rdf:Alt></dc:title>\n"));
            // Also set photoshop:Headline
            remove_xml_element(&mut result, "photoshop:Headline");
            new_elements.push_str(&format!("  <photoshop:Headline>{t_esc}</photoshop:Headline>\n"));
        }

        if let Some(d) = description {
            let d_esc = xml_escape(d);
            remove_xml_element(&mut result, "dc:description");
            new_elements.push_str(&format!("  <dc:description><rdf:Alt><rdf:li xml:lang=\"x-default\">{d_esc}</rdf:li></rdf:Alt></dc:description>\n"));
        }

        if let Some(kw) = keywords {
            remove_xml_element(&mut result, "dc:subject");
            new_elements.push_str("  <dc:subject><rdf:Bag>\n");
            for k in kw {
                new_elements.push_str(&format!("    <rdf:li>{}</rdf:li>\n", xml_escape(k)));
            }
            new_elements.push_str("  </rdf:Bag></dc:subject>\n");
        }

        // Re-find position after removals
        if let Some(pos) = result.find("</rdf:Description>") {
            result.insert_str(pos, &new_elements);
        }
    }

    result
}

/// Remove an XML element and its contents from a string.
fn remove_xml_element(xml: &mut String, tag: &str) {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    if let Some(start) = xml.find(&open) {
        if let Some(end) = xml[start..].find(&close) {
            let end_abs = start + end + close.len();
            // Also remove trailing newline if present
            let end_abs = if xml.as_bytes().get(end_abs) == Some(&b'\n') {
                end_abs + 1
            } else {
                end_abs
            };
            xml.replace_range(start..end_abs, "");
        }
    }
}

/// Escape special XML characters.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
     .replace('\'', "&apos;")
}

// ============================================================================
// IPTC-IIM Metadata Writing (APP13 / Photoshop 3.0)
// ============================================================================

const IPTC_HEADER: &[u8] = b"Photoshop 3.0\0";
const IPTC_8BIM: &[u8] = b"8BIM";

/// Update or create IPTC metadata in the JPEG.
/// Writes IPTC caption (2:120) and keywords (2:25) for broad tool compatibility.
fn update_iptc_metadata(
    jpeg: &mut Jpeg,
    ai_result: &AiResult,
    existing: &ExifData,
    fields: &ExifFields,
) {
    let title = if fields.write_title {
        ai_result.title.as_ref()
            .filter(|_| existing.title.is_none() || fields.overwrite_existing)
    } else { None };

    let description = if fields.write_description {
        ai_result.description.as_ref()
            .filter(|_| existing.description.is_none() || fields.overwrite_existing)
    } else { None };

    let keywords: Option<&Vec<String>> = if fields.write_tags {
        ai_result.tags.as_ref()
            .filter(|_| existing.keywords.is_none() || fields.overwrite_existing)
    } else { None };

    if title.is_none() && description.is_none() && keywords.is_none() {
        return;
    }

    // Find existing APP13 segment
    let iptc_pos = jpeg.segments().iter().position(|s| {
        s.marker() == 0xED && s.contents().starts_with(IPTC_HEADER)
    });

    // Read existing IPTC data (preserve non-AI records)
    let existing_iptc = iptc_pos.map(|pos| {
        jpeg.segments()[pos].contents().to_vec()
    });

    // Build new IPTC APP13 contents
    let new_contents = build_iptc_contents(
        existing_iptc.as_deref(),
        title.map(|s| s.as_str()),
        description.map(|s| s.as_str()),
        keywords,
    );

    let new_segment = JpegSegment::new_with_contents(0xED, Bytes::from(new_contents));

    let segments = jpeg.segments_mut();
    if let Some(pos) = iptc_pos {
        segments[pos] = new_segment;
    } else {
        // Insert after XMP or EXIF
        let insert_pos = segments.len().min(4);
        segments.insert(insert_pos, new_segment);
    }
}

/// Build IPTC APP13 segment contents.
/// Preserves existing 8BIM resources, adds/replaces IPTC-IIM resource (0x0404).
fn build_iptc_contents(
    existing: Option<&[u8]>,
    title: Option<&str>,
    description: Option<&str>,
    keywords: Option<&Vec<String>>,
) -> Vec<u8> {
    let mut result = Vec::new();
    result.extend_from_slice(IPTC_HEADER);

    // Copy existing 8BIM resources except IPTC-IIM (0x0404)
    if let Some(data) = existing {
        let mut pos = IPTC_HEADER.len();
        while pos + 12 <= data.len() {
            if &data[pos..pos + 4] != IPTC_8BIM {
                break;
            }
            let resource_id = u16::from_be_bytes([data[pos + 4], data[pos + 5]]);
            // Skip pascal string (1 byte length + string + padding to even)
            let pascal_len = data[pos + 6] as usize;
            let pascal_padded = if (pascal_len + 1) % 2 == 0 { pascal_len + 1 } else { pascal_len + 2 };
            let data_start = pos + 6 + pascal_padded;
            if data_start + 4 > data.len() { break; }
            let data_len = u32::from_be_bytes([
                data[data_start], data[data_start + 1],
                data[data_start + 2], data[data_start + 3],
            ]) as usize;
            let resource_end = data_start + 4 + data_len;
            let resource_end_padded = if data_len % 2 == 0 { resource_end } else { resource_end + 1 };

            if resource_id != 0x0404 {
                // Preserve this resource
                let end = resource_end_padded.min(data.len());
                result.extend_from_slice(&data[pos..end]);
            }

            pos = resource_end_padded;
        }
    }

    // Build IPTC-IIM dataset records
    let mut iptc_data = Vec::new();

    // Record version (2:0) — required
    iptc_data.extend_from_slice(&[0x1C, 0x02, 0x00, 0x00, 0x02, 0x00, 0x02]);

    // Object Name / Title (2:5)
    if let Some(t) = title {
        let bytes = t.as_bytes();
        let len = bytes.len().min(64) as u16;
        iptc_data.extend_from_slice(&[0x1C, 0x02, 0x05]);
        iptc_data.extend_from_slice(&len.to_be_bytes());
        iptc_data.extend_from_slice(&bytes[..len as usize]);
    }

    // Keywords (2:25) — one record per keyword
    if let Some(kw) = keywords {
        for k in kw {
            let bytes = k.as_bytes();
            let len = bytes.len().min(64) as u16;
            iptc_data.extend_from_slice(&[0x1C, 0x02, 0x19]);
            iptc_data.extend_from_slice(&len.to_be_bytes());
            iptc_data.extend_from_slice(&bytes[..len as usize]);
        }
    }

    // Caption/Abstract (2:120)
    if let Some(d) = description {
        let bytes = d.as_bytes();
        let len = bytes.len().min(2000) as u16;
        iptc_data.extend_from_slice(&[0x1C, 0x02, 0x78]);
        iptc_data.extend_from_slice(&len.to_be_bytes());
        iptc_data.extend_from_slice(&bytes[..len as usize]);
    }

    // Write the IPTC-IIM as 8BIM resource 0x0404
    if !iptc_data.is_empty() {
        result.extend_from_slice(IPTC_8BIM);
        result.extend_from_slice(&0x0404u16.to_be_bytes()); // resource ID
        result.push(0x00); // pascal string (empty, length 0)
        result.push(0x00); // padding to even
        let data_len = iptc_data.len() as u32;
        result.extend_from_slice(&data_len.to_be_bytes());
        result.extend_from_slice(&iptc_data);
        if iptc_data.len() % 2 != 0 {
            result.push(0x00); // pad to even
        }
    }

    result
}

/// A raw IFD entry to inject into a TIFF, built in the correct endianness.
struct RawIfdEntry {
    tag_id: u16,
    data_format: u16,   // TIFF data format (2=ASCII, 1=BYTE, etc.)
    count: u32,
    inline_value: [u8; 4],  // value if data fits in 4 bytes
    extra_data: Option<Vec<u8>>,  // data if > 4 bytes
}

/// Build a raw IFD entry for a string tag (ASCII, format=2).
fn make_string_entry(tag_id: u16, value: &str, _big_endian: bool) -> RawIfdEntry {
    let mut data = value.as_bytes().to_vec();
    data.push(0); // null terminator
    let count = data.len() as u32;

    let (inline_value, extra_data) = if data.len() <= 4 {
        let mut inline = [0u8; 4];
        inline[..data.len()].copy_from_slice(&data);
        (inline, None)
    } else {
        ([0u8; 4], Some(data))
    };

    RawIfdEntry { tag_id, data_format: 2, count, inline_value, extra_data }
}

/// Build a raw IFD entry for a UTF-16LE byte tag (XP* tags, format=1 BYTE).
fn make_xp_entry(tag_id: u16, value: &str) -> RawIfdEntry {
    let data = encode_utf16le(value);
    let count = data.len() as u32;

    let (inline_value, extra_data) = if data.len() <= 4 {
        let mut inline = [0u8; 4];
        inline[..data.len()].copy_from_slice(&data);
        (inline, None)
    } else {
        ([0u8; 4], Some(data))
    };

    RawIfdEntry { tag_id, data_format: 1, count, inline_value, extra_data }
}

/// Build a raw IFD entry for UserComment (UNDEFINED format=7, with ASCII prefix).
fn make_user_comment_entry(tag_id: u16, value: &str) -> RawIfdEntry {
    let mut data = b"ASCII\0\0\0".to_vec();
    data.extend_from_slice(value.as_bytes());
    let count = data.len() as u32;

    RawIfdEntry {
        tag_id,
        data_format: 7, // UNDEFINED
        count,
        inline_value: [0u8; 4],
        extra_data: Some(data),
    }
}

/// Inject AI-generated tags directly into the original raw TIFF data,
/// building IFD entries in the correct endianness.
/// Writes IFD0 tags (ImageDescription, XP*) and ExifIFD tags (UserComment)
/// to their correct sub-IFDs.
fn inject_ai_tags_into_tiff(
    original: &[u8],
    ai_result: &AiResult,
    existing: &ExifData,
    fields: &ExifFields,
) -> Result<Vec<u8>> {
    if original.len() < 8 {
        anyhow::bail!("Original TIFF data too short");
    }

    let big_endian = match &original[0..2] {
        b"MM" => true,
        b"II" => false,
        _ => anyhow::bail!("Invalid TIFF byte order"),
    };

    // Build IFD0 entries and ExifIFD entries separately
    let mut ifd0_entries: Vec<RawIfdEntry> = Vec::new();
    let mut exif_ifd_entries: Vec<RawIfdEntry> = Vec::new();

    if fields.write_title {
        if let Some(ref title) = ai_result.title {
            if existing.title.is_none() || fields.overwrite_existing {
                ifd0_entries.push(make_string_entry(0x010E, title, big_endian)); // ImageDescription
                ifd0_entries.push(make_xp_entry(TAG_XP_TITLE, title));
            }
        }
    }

    if fields.write_description {
        if let Some(ref desc) = ai_result.description {
            if existing.description.is_none() || fields.overwrite_existing {
                exif_ifd_entries.push(make_user_comment_entry(0x9286, desc)); // UserComment → ExifIFD
                ifd0_entries.push(make_xp_entry(TAG_XP_COMMENT, desc));
            }
        }
    }

    if fields.write_tags {
        if let Some(ref tags) = ai_result.tags {
            if existing.keywords.is_none() || fields.overwrite_existing {
                let kw = tags.join("; ");
                ifd0_entries.push(make_xp_entry(TAG_XP_KEYWORDS, &kw));
            }
        }
    }

    if fields.write_subject {
        if let Some(ref subjects) = ai_result.subject {
            if !subjects.is_empty() && (existing.subject.is_none() || fields.overwrite_existing) {
                let subj = subjects.join("; ");
                ifd0_entries.push(make_xp_entry(TAG_XP_SUBJECT, &subj));
            }
        }
    }

    if ifd0_entries.is_empty() && exif_ifd_entries.is_empty() {
        return Ok(original.to_vec());
    }

    // Helper closures for reading/writing in the original's endianness
    let read_u16 = |data: &[u8], offset: usize| -> u16 {
        if big_endian {
            u16::from_be_bytes([data[offset], data[offset + 1]])
        } else {
            u16::from_le_bytes([data[offset], data[offset + 1]])
        }
    };
    let read_u32 = |data: &[u8], offset: usize| -> u32 {
        if big_endian {
            u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
        } else {
            u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
        }
    };
    let encode_u16 = |val: u16| -> [u8; 2] {
        if big_endian { val.to_be_bytes() } else { val.to_le_bytes() }
    };
    let encode_u32 = |val: u32| -> [u8; 4] {
        if big_endian { val.to_be_bytes() } else { val.to_le_bytes() }
    };

    // Parse IFD0
    let ifd0_offset = read_u32(original, 4) as usize;
    if ifd0_offset + 2 > original.len() {
        anyhow::bail!("IFD0 offset out of bounds");
    }
    let ifd0_count = read_u16(original, ifd0_offset) as usize;
    let ifd0_start = ifd0_offset + 2;
    let ifd0_end = ifd0_start + ifd0_count * 12;
    if ifd0_end + 4 > original.len() {
        anyhow::bail!("IFD0 entries extend beyond TIFF data");
    }

    let ifd0_tag_ids: Vec<u16> = (0..ifd0_count)
        .map(|i| read_u16(original, ifd0_start + i * 12))
        .collect();
    let ifd0_next = read_u32(original, ifd0_end);

    // Find ExifIFD offset from IFD0 (tag 0x8769)
    let exif_ifd_offset: Option<usize> = ifd0_tag_ids.iter().enumerate()
        .find(|&(_, t)| *t == 0x8769)
        .map(|(i, _)| {
            let eo = ifd0_start + i * 12;
            read_u32(original, eo + 8) as usize
        });

    // Parse ExifIFD if it exists
    let (exif_count, exif_start, _exif_end, exif_tag_ids, exif_next) = if let Some(eo) = exif_ifd_offset {
        if eo + 2 <= original.len() {
            let count = read_u16(original, eo) as usize;
            let start = eo + 2;
            let end = start + count * 12;
            if end + 4 <= original.len() {
                let tags: Vec<u16> = (0..count)
                    .map(|i| read_u16(original, start + i * 12))
                    .collect();
                let next = read_u32(original, end);
                (count, start, end, tags, next)
            } else {
                (0, 0, 0, Vec::new(), 0u32)
            }
        } else {
            (0, 0, 0, Vec::new(), 0u32)
        }
    } else {
        (0, 0, 0, Vec::new(), 0u32)
    };

    let mut result = original.to_vec();

    // === Rebuild ExifIFD at the end (if we have ExifIFD entries to add) ===
    let new_exif_ifd_start: Option<u32> = if !exif_ifd_entries.is_empty() && exif_ifd_offset.is_some() {
        let exif_append_count = exif_ifd_entries.iter()
            .filter(|e| !exif_tag_ids.contains(&e.tag_id)).count();
        let total = exif_count + exif_append_count;

        let start = result.len() as u32;

        // Entry count
        result.extend_from_slice(&encode_u16(total as u16));

        // Copy original ExifIFD entries
        for i in 0..exif_count {
            let eo = exif_start + i * 12;
            result.extend_from_slice(&original[eo..eo + 12]);
        }

        // Placeholder slots for new entries
        let append_start = result.len();
        for _ in 0..exif_append_count {
            result.extend_from_slice(&[0u8; 12]);
        }

        // Next-IFD pointer
        result.extend_from_slice(&encode_u32(exif_next));

        // Append data blobs and build entries
        let mut data_off = result.len() as u32;
        let mut raw: Vec<(u16, [u8; 12])> = Vec::new();
        for entry in &exif_ifd_entries {
            let mut ib = [0u8; 12];
            ib[0..2].copy_from_slice(&encode_u16(entry.tag_id));
            ib[2..4].copy_from_slice(&encode_u16(entry.data_format));
            ib[4..8].copy_from_slice(&encode_u32(entry.count));
            if let Some(ref extra) = entry.extra_data {
                ib[8..12].copy_from_slice(&encode_u32(data_off));
                result.extend_from_slice(extra);
                data_off += extra.len() as u32;
            } else {
                ib[8..12].copy_from_slice(&entry.inline_value);
            }
            raw.push((entry.tag_id, ib));
        }

        // Fill entries
        let entries_base = start as usize + 2;
        let mut slot = 0;
        for (tag_id, ib) in &raw {
            if let Some(idx) = exif_tag_ids.iter().position(|&t| t == *tag_id) {
                let off = entries_base + idx * 12;
                result[off..off + 12].copy_from_slice(ib);
            } else {
                let off = append_start + slot * 12;
                result[off..off + 12].copy_from_slice(ib);
                slot += 1;
            }
        }

        Some(start)
    } else {
        None
    };

    // === Rebuild IFD0 at the end ===
    let ifd0_append_count = ifd0_entries.iter()
        .filter(|e| !ifd0_tag_ids.contains(&e.tag_id)).count();
    let ifd0_total = ifd0_count + ifd0_append_count;

    let new_ifd0_start = result.len() as u32;

    // Entry count
    result.extend_from_slice(&encode_u16(ifd0_total as u16));

    // Copy original IFD0 entries
    for i in 0..ifd0_count {
        let eo = ifd0_start + i * 12;
        result.extend_from_slice(&original[eo..eo + 12]);
    }

    // Placeholder slots
    let ifd0_append_start = result.len();
    for _ in 0..ifd0_append_count {
        result.extend_from_slice(&[0u8; 12]);
    }

    // Next-IFD pointer
    result.extend_from_slice(&encode_u32(ifd0_next));

    // Append data blobs and build entries
    let mut data_off = result.len() as u32;
    let mut raw: Vec<(u16, [u8; 12])> = Vec::new();
    for entry in &ifd0_entries {
        let mut ib = [0u8; 12];
        ib[0..2].copy_from_slice(&encode_u16(entry.tag_id));
        ib[2..4].copy_from_slice(&encode_u16(entry.data_format));
        ib[4..8].copy_from_slice(&encode_u32(entry.count));
        if let Some(ref extra) = entry.extra_data {
            ib[8..12].copy_from_slice(&encode_u32(data_off));
            result.extend_from_slice(extra);
            data_off += extra.len() as u32;
        } else {
            ib[8..12].copy_from_slice(&entry.inline_value);
        }
        raw.push((entry.tag_id, ib));
    }

    // Fill entries
    let ifd0_entries_base = new_ifd0_start as usize + 2;
    let mut slot = 0;
    for (tag_id, ib) in &raw {
        if let Some(idx) = ifd0_tag_ids.iter().position(|&t| t == *tag_id) {
            let off = ifd0_entries_base + idx * 12;
            result[off..off + 12].copy_from_slice(ib);
        } else {
            let off = ifd0_append_start + slot * 12;
            result[off..off + 12].copy_from_slice(ib);
            slot += 1;
        }
    }

    // If we rebuilt ExifIFD, update the ExifIFD pointer in the new IFD0
    if let Some(new_exif_off) = new_exif_ifd_start {
        // Find the ExifIFD pointer entry (0x8769) in the new IFD0 and update it
        for i in 0..ifd0_total {
            let eo = ifd0_entries_base + i * 12;
            if eo + 12 <= result.len() {
                let tag = read_u16(&result, eo);
                if tag == 0x8769 {
                    result[eo + 8..eo + 12].copy_from_slice(&encode_u32(new_exif_off));
                    break;
                }
            }
        }
    }

    // Update TIFF header to point to new IFD0
    result[4..8].copy_from_slice(&encode_u32(new_ifd0_start));

    Ok(result)
}

/// Collect GPS tags into the tag list.
fn collect_gps_tags(tags: &mut Vec<ExifTag>, gps: &GpsCoords) {
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

    if let Ok(tag) = ExifTag::from_u16_with_data(
        TAG_GPS_LATITUDE_REF,
        &ExifTagFormat::STRING,
        &format!("{lat_ref}\0").into_bytes(),
        &Endian::Little,
        &ExifTagGroup::GPSIFD,
    ) {
        tags.push(tag);
    }

    let lat_bytes = encode_gps_rational(lat_deg, lat_min, lat_sec, 10000);
    if let Ok(tag) = ExifTag::from_u16_with_data(
        TAG_GPS_LATITUDE,
        &ExifTagFormat::RATIONAL64U,
        &lat_bytes,
        &Endian::Little,
        &ExifTagGroup::GPSIFD,
    ) {
        tags.push(tag);
    }

    if let Ok(tag) = ExifTag::from_u16_with_data(
        TAG_GPS_LONGITUDE_REF,
        &ExifTagFormat::STRING,
        &format!("{lon_ref}\0").into_bytes(),
        &Endian::Little,
        &ExifTagGroup::GPSIFD,
    ) {
        tags.push(tag);
    }

    let lon_bytes = encode_gps_rational(lon_deg, lon_min, lon_sec, 10000);
    if let Ok(tag) = ExifTag::from_u16_with_data(
        TAG_GPS_LONGITUDE,
        &ExifTagFormat::RATIONAL64U,
        &lon_bytes,
        &Endian::Little,
        &ExifTagGroup::GPSIFD,
    ) {
        tags.push(tag);
    }
}
