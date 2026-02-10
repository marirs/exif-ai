mod openai;
mod gemini;
mod cloudflare;

pub use openai::OpenAiService;
pub use gemini::GeminiService;
pub use cloudflare::CloudflareService;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// The structured data returned by AI vision analysis.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AiResult {
    pub title: Option<String>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub gps: Option<GpsCoords>,
    pub subject: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpsCoords {
    pub latitude: f64,
    pub longitude: f64,
}

/// Trait for AI vision services.
#[async_trait::async_trait]
pub trait AiService: Send + Sync {
    fn name(&self) -> &str;
    async fn analyze(&self, image_base64: &str, prompt: &str, mime_type: &str) -> Result<AiResult>;
}

/// Build the AI prompt that asks for structured JSON output.
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

/// Parse the AI response text into an AiResult.
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
