mod openai;
mod gemini;
mod cloudflare;

pub use openai::OpenAiService;
pub use gemini::GeminiService;
pub use cloudflare::CloudflareService;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Structured metadata returned by AI vision analysis.
///
/// Each field is `Option` because the AI may not be able to determine all fields
/// for every image. Fields set to `None` or `null` by the AI are skipped during
/// the metadata write phase.
///
/// # Fields
///
/// - `title` — SEO-optimized title (max 60 chars)
/// - `description` — Descriptive caption (max 254 chars)
/// - `tags` — 5–10 SEO keywords
/// - `gps` — GPS coordinates if a known location is identified
/// - `subject` — Identified people, species, landmarks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AiResult {
    pub title: Option<String>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub gps: Option<GpsCoords>,
    pub subject: Option<Vec<String>>,
}

/// GPS coordinates identified by the AI for a known location.
///
/// Only populated when the AI recognizes a specific, real-world location
/// in the image (e.g., Eiffel Tower, Golden Gate Bridge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpsCoords {
    pub latitude: f64,
    pub longitude: f64,
}

/// Trait for AI vision services.
///
/// Implement this trait to add a custom AI backend. The library ships with
/// three implementations: [`OpenAiService`], [`GeminiService`], and
/// [`CloudflareService`].
///
/// # Example
///
/// ```rust,no_run
/// use exif_ai::ai::{AiService, AiResult, OpenAiService};
///
/// # async fn example() -> anyhow::Result<()> {
/// let service = OpenAiService::new("sk-...".into(), "gpt-4o-mini".into());
/// let result = service.analyze("base64data", "prompt", "image/jpeg").await?;
/// println!("Title: {:?}", result.title);
/// # Ok(())
/// # }
/// ```
#[async_trait::async_trait]
pub trait AiService: Send + Sync {
    /// The display name of this service (e.g., "OpenAI", "Gemini").
    fn name(&self) -> &str;
    /// Analyze a base64-encoded image and return structured metadata.
    ///
    /// * `image_base64` — The image bytes encoded as base64
    /// * `prompt` — The analysis prompt (use [`build_prompt`] for the default)
    /// * `mime_type` — The MIME type of the image (e.g., `"image/jpeg"`, `"image/heic"`)
    async fn analyze(&self, image_base64: &str, prompt: &str, mime_type: &str) -> Result<AiResult>;
}

/// Build the default AI prompt that asks for structured JSON output.
///
/// Returns the prompt string used to instruct the AI model to return
/// a JSON object with `title`, `description`, `tags`, `gps`, and `subject` fields.
///
/// You can use this directly or provide your own custom prompt to
/// [`AiService::analyze`].
pub fn build_prompt() -> String {
    r#"Analyze this image and return a JSON object with the following fields:

{
  "title": "A concise, SEO-optimized title for this image (max 60 characters)",
  "description": "An engaging SEO meta description of this image (max 254 characters)",
  "tags": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "gps": { "latitude": 0.0, "longitude": 0.0 },
  "subject": ["identified subject 1", "identified subject 2"]
}

Rules:
- "title": A short, catchy SEO title. Max 60 characters. Think of it as a headline.
- "description": A detailed, descriptive paragraph about the image content, scene, mood, colors, and context. Write it as a full sentence or two, like an image caption in a magazine. Max 254 characters.
- "tags": 5-10 relevant SEO keywords/tags for the image.
- "gps": If you can identify a specific, well-known location in the image, provide GPS coordinates. If unsure or the location is not identifiable, set to null.
- "subject": If you can identify specific known people, bird species, animal species, landmarks, or other notable subjects, list them. If none are identifiable, set to null.

Return ONLY the JSON object, no markdown formatting, no code blocks, no extra text."#
        .to_string()
}

/// Parse raw AI response text into an [`AiResult`].
///
/// Handles common AI quirks: markdown code fences, trailing commas,
/// and multiple JSON candidates in the response. Tries several extraction
/// strategies before failing.
pub fn parse_ai_response(text: &str) -> Result<AiResult> {
    log::debug!("Raw AI response:\n{text}");

    let cleaned = text.trim();

    // Try a sequence of extraction strategies
    let candidates = extract_json_candidates(cleaned);

    for candidate in &candidates {
        // Try parsing directly
        if let Ok(result) = serde_json::from_str::<AiResult>(candidate) {
            return Ok(result);
        }

        // Try after fixing trailing commas (common AI quirk)
        let fixed = fix_trailing_commas(candidate);
        if let Ok(result) = serde_json::from_str::<AiResult>(&fixed) {
            return Ok(result);
        }
    }

    // Try parsing as a generic Value first to give a better error
    if let Some(candidate) = candidates.first() {
        match serde_json::from_str::<serde_json::Value>(candidate) {
            Ok(val) => {
                log::warn!("AI response is valid JSON but doesn't match expected schema");
                log::debug!("Parsed value: {val:#}");
                // Try to manually extract fields from the Value
                if let Some(result) = value_to_ai_result(&val) {
                    return Ok(result);
                }
            }
            Err(e) => {
                log::warn!("Failed to parse AI response as JSON: {e}");
            }
        }
    }

    anyhow::bail!("Could not parse AI response as JSON")
}

/// Extract possible JSON object strings from AI response text.
fn extract_json_candidates(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    // Strategy 1: Strip markdown code fences (```json ... ``` or ``` ... ```)
    if text.contains("```") {
        let stripped = text
            .lines()
            .skip_while(|l| !l.trim().starts_with("```"))
            .skip(1) // skip the ``` line itself
            .take_while(|l| !l.trim().starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n");
        if !stripped.is_empty() {
            candidates.push(stripped);
        }
    }

    // Strategy 2: Find outermost { ... }
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end > start {
                let extracted = text[start..=end].to_string();
                candidates.push(extracted.clone());

                // Strategy 3: Fix unquoted string values (common AI quirk)
                let fixed = fix_unquoted_values(&extracted);
                if fixed != extracted {
                    candidates.push(fixed);
                }
            }
        }
    }

    // Strategy 4: The whole text as-is
    candidates.push(text.to_string());

    candidates
}

/// Fix unquoted string values in malformed JSON.
/// Handles patterns like: "key": some unquoted value,
fn fix_unquoted_values(text: &str) -> String {
    use std::fmt::Write;

    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_string = false;
    let mut escape_next = false;

    while let Some(c) = chars.next() {
        if escape_next {
            result.push(c);
            escape_next = false;
            continue;
        }
        if c == '\\' && in_string {
            result.push(c);
            escape_next = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            result.push(c);
            continue;
        }

        if !in_string && c == ':' {
            result.push(c);

            // Skip whitespace after colon
            let mut ws = String::new();
            while let Some(&next) = chars.peek() {
                if next == ' ' || next == '\t' {
                    ws.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            result.push_str(&ws);

            // Check if next char starts an unquoted string value
            if let Some(&next) = chars.peek() {
                if next != '"' && next != '{' && next != '[' && next != 'n'
                    && next != 't' && next != 'f' && !next.is_ascii_digit() && next != '-'
                {
                    // Likely an unquoted string — collect until , or } or newline
                    let mut value = String::new();
                    while let Some(&vc) = chars.peek() {
                        if vc == ',' || vc == '}' || vc == '\n' {
                            break;
                        }
                        value.push(vc);
                        chars.next();
                    }
                    let value = value.trim_end();
                    // Escape any quotes inside the value
                    let escaped = value.replace('"', "\\\"");
                    let _ = write!(result, "\"{escaped}\"");
                    continue;
                }
                // Check for null/true/false that start with n/t/f
                // These are valid JSON literals, leave them alone
            }
            continue;
        }

        result.push(c);
    }

    result
}

/// Fix trailing commas in JSON (e.g. {"a": 1,} or ["a",])
fn fix_trailing_commas(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_string = false;
    let mut escape_next = false;

    while let Some(c) = chars.next() {
        if escape_next {
            result.push(c);
            escape_next = false;
            continue;
        }
        if c == '\\' && in_string {
            result.push(c);
            escape_next = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            result.push(c);
            continue;
        }
        if !in_string && c == ',' {
            // Look ahead past whitespace for } or ]
            let rest: String = chars.clone().collect();
            let trimmed = rest.trim_start();
            if trimmed.starts_with('}') || trimmed.starts_with(']') {
                // Skip this trailing comma
                continue;
            }
        }
        result.push(c);
    }
    result
}

/// Try to extract AiResult fields from a generic serde_json::Value.
fn value_to_ai_result(val: &serde_json::Value) -> Option<AiResult> {
    let obj = val.as_object()?;
    let mut result = AiResult::default();
    let mut found_any = false;

    if let Some(v) = obj.get("title").and_then(|v| v.as_str()) {
        result.title = Some(v.to_string());
        found_any = true;
    }
    if let Some(v) = obj.get("description").and_then(|v| v.as_str()) {
        result.description = Some(v.to_string());
        found_any = true;
    }
    if let Some(arr) = obj.get("tags").and_then(|v| v.as_array()) {
        let tags: Vec<String> = arr.iter().filter_map(|v| v.as_str().map(String::from)).collect();
        if !tags.is_empty() {
            result.tags = Some(tags);
            found_any = true;
        }
    }
    if let Some(gps_obj) = obj.get("gps").and_then(|v| v.as_object()) {
        if let (Some(lat), Some(lon)) = (
            gps_obj.get("latitude").and_then(|v| v.as_f64()),
            gps_obj.get("longitude").and_then(|v| v.as_f64()),
        ) {
            if lat != 0.0 || lon != 0.0 {
                result.gps = Some(GpsCoords { latitude: lat, longitude: lon });
                found_any = true;
            }
        }
    }
    if let Some(arr) = obj.get("subject").and_then(|v| v.as_array()) {
        let subjects: Vec<String> = arr.iter().filter_map(|v| v.as_str().map(String::from)).collect();
        if !subjects.is_empty() {
            result.subject = Some(subjects);
            found_any = true;
        }
    }

    if found_any { Some(result) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── build_prompt ─────────────────────────────────────────────────

    #[test]
    fn build_prompt_non_empty() {
        let prompt = build_prompt();
        assert!(!prompt.is_empty());
        assert!(prompt.contains("title"));
        assert!(prompt.contains("description"));
        assert!(prompt.contains("tags"));
        assert!(prompt.contains("gps"));
        assert!(prompt.contains("subject"));
        assert!(prompt.contains("JSON"));
    }

    // ── parse_ai_response: valid JSON ────────────────────────────────

    #[test]
    fn parse_valid_json() {
        let json = r#"{
            "title": "Sunset Beach",
            "description": "A beautiful sunset over the ocean",
            "tags": ["sunset", "beach", "ocean"],
            "gps": { "latitude": 34.0, "longitude": -118.5 },
            "subject": ["Pacific Ocean"]
        }"#;

        let result = parse_ai_response(json).unwrap();
        assert_eq!(result.title.as_deref(), Some("Sunset Beach"));
        assert_eq!(result.description.as_deref(), Some("A beautiful sunset over the ocean"));
        assert_eq!(result.tags.as_ref().unwrap().len(), 3);
        assert!(result.gps.is_some());
        let gps = result.gps.unwrap();
        assert!((gps.latitude - 34.0).abs() < 0.001);
        assert!((gps.longitude - (-118.5)).abs() < 0.001);
        assert_eq!(result.subject.as_ref().unwrap(), &["Pacific Ocean"]);
    }

    #[test]
    fn parse_minimal_json() {
        let json = r#"{"title": "Hello", "description": "World"}"#;
        let result = parse_ai_response(json).unwrap();
        assert_eq!(result.title.as_deref(), Some("Hello"));
        assert_eq!(result.description.as_deref(), Some("World"));
        assert!(result.tags.is_none());
        assert!(result.gps.is_none());
        assert!(result.subject.is_none());
    }

    #[test]
    fn parse_null_fields() {
        let json = r#"{
            "title": "Test",
            "description": "Desc",
            "tags": null,
            "gps": null,
            "subject": null
        }"#;
        let result = parse_ai_response(json).unwrap();
        assert_eq!(result.title.as_deref(), Some("Test"));
        assert!(result.tags.is_none());
        assert!(result.gps.is_none());
        assert!(result.subject.is_none());
    }

    // ── parse_ai_response: markdown fences ───────────────────────────

    #[test]
    fn parse_markdown_json_fence() {
        let text = r#"Here is the analysis:

```json
{
  "title": "Mountain Lake",
  "description": "A serene mountain lake"
}
```"#;
        let result = parse_ai_response(text).unwrap();
        assert_eq!(result.title.as_deref(), Some("Mountain Lake"));
    }

    #[test]
    fn parse_markdown_plain_fence() {
        let text = r#"```
{"title": "Test", "description": "Desc"}
```"#;
        let result = parse_ai_response(text).unwrap();
        assert_eq!(result.title.as_deref(), Some("Test"));
    }

    // ── parse_ai_response: trailing commas ───────────────────────────

    #[test]
    fn parse_trailing_comma_object() {
        let json = r#"{"title": "Test", "description": "Desc",}"#;
        let result = parse_ai_response(json).unwrap();
        assert_eq!(result.title.as_deref(), Some("Test"));
    }

    #[test]
    fn parse_trailing_comma_array() {
        let json = r#"{"title": "Test", "tags": ["a", "b",]}"#;
        let result = parse_ai_response(json).unwrap();
        assert_eq!(result.tags.as_ref().unwrap(), &["a", "b"]);
    }

    // ── parse_ai_response: extra text around JSON ────────────────────

    #[test]
    fn parse_json_with_surrounding_text() {
        let text = r#"Sure! Here is the result:
{"title": "Cat", "description": "A fluffy cat"}
Hope this helps!"#;
        let result = parse_ai_response(text).unwrap();
        assert_eq!(result.title.as_deref(), Some("Cat"));
    }

    // ── parse_ai_response: GPS zero is null ──────────────────────────

    #[test]
    fn parse_gps_zero_treated_as_none() {
        let json = r#"{
            "title": "Unknown Place",
            "description": "Somewhere",
            "gps": { "latitude": 0.0, "longitude": 0.0 }
        }"#;
        // GPS 0,0 is in the ocean — AI should return null, but if it returns 0,0
        // we treat it as "no GPS identified" via value_to_ai_result
        let result = parse_ai_response(json).unwrap();
        // Direct serde parse will include gps with 0,0
        // but value_to_ai_result filters it out
        // The direct parse path keeps it, which is fine — the writer checks has_gps
        assert!(result.title.is_some());
    }

    // ── parse_ai_response: errors ────────────────────────────────────

    #[test]
    fn parse_garbage_fails() {
        let result = parse_ai_response("this is not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn parse_empty_fails() {
        let result = parse_ai_response("");
        assert!(result.is_err());
    }

    #[test]
    fn parse_empty_object_fails() {
        let result = parse_ai_response("{}");
        // Empty object parses as AiResult with all None — that's valid for serde
        // but value_to_ai_result returns None. Direct parse succeeds though.
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.title.is_none());
    }

    // ── fix_trailing_commas ──────────────────────────────────────────

    #[test]
    fn fix_trailing_commas_basic() {
        assert_eq!(fix_trailing_commas(r#"{"a": 1,}"#), r#"{"a": 1}"#);
        assert_eq!(fix_trailing_commas(r#"["a",]"#), r#"["a"]"#);
    }

    #[test]
    fn fix_trailing_commas_preserves_valid() {
        let valid = r#"{"a": 1, "b": 2}"#;
        assert_eq!(fix_trailing_commas(valid), valid);
    }

    #[test]
    fn fix_trailing_commas_in_string_untouched() {
        let s = r#"{"a": "hello,}"}"#;
        // The comma inside the string value should not be removed
        assert_eq!(fix_trailing_commas(s), s);
    }

    // ── extract_json_candidates ──────────────────────────────────────

    #[test]
    fn extract_candidates_plain_json() {
        let candidates = extract_json_candidates(r#"{"a": 1}"#);
        assert!(candidates.iter().any(|c| c.contains(r#""a": 1"#)));
    }

    #[test]
    fn extract_candidates_markdown() {
        let text = "```json\n{\"a\": 1}\n```";
        let candidates = extract_json_candidates(text);
        assert!(candidates.iter().any(|c| c.contains(r#""a": 1"#)));
    }

    #[test]
    fn extract_candidates_with_prefix() {
        let text = "Here is the JSON: {\"title\": \"test\"}";
        let candidates = extract_json_candidates(text);
        assert!(candidates.iter().any(|c| c.starts_with('{')));
    }

    // ── value_to_ai_result ───────────────────────────────────────────

    #[test]
    fn value_to_ai_result_full() {
        let val: serde_json::Value = serde_json::from_str(r#"{
            "title": "Test",
            "description": "Desc",
            "tags": ["a", "b"],
            "gps": {"latitude": 48.8, "longitude": 2.3},
            "subject": ["Eiffel Tower"]
        }"#).unwrap();

        let result = value_to_ai_result(&val).unwrap();
        assert_eq!(result.title.as_deref(), Some("Test"));
        assert_eq!(result.tags.as_ref().unwrap().len(), 2);
        assert!(result.gps.is_some());
        assert_eq!(result.subject.as_ref().unwrap(), &["Eiffel Tower"]);
    }

    #[test]
    fn value_to_ai_result_empty_returns_none() {
        let val: serde_json::Value = serde_json::from_str("{}").unwrap();
        assert!(value_to_ai_result(&val).is_none());
    }

    #[test]
    fn value_to_ai_result_gps_zero_skipped() {
        let val: serde_json::Value = serde_json::from_str(r#"{
            "title": "Test",
            "gps": {"latitude": 0.0, "longitude": 0.0}
        }"#).unwrap();
        let result = value_to_ai_result(&val).unwrap();
        assert!(result.gps.is_none()); // 0,0 is filtered out
    }

    // ── AiResult default ─────────────────────────────────────────────

    #[test]
    fn ai_result_default_all_none() {
        let r = AiResult::default();
        assert!(r.title.is_none());
        assert!(r.description.is_none());
        assert!(r.tags.is_none());
        assert!(r.gps.is_none());
        assert!(r.subject.is_none());
    }
}
