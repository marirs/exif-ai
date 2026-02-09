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
    async fn analyze(&self, image_base64: &str, prompt: &str) -> Result<AiResult>;
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
- "title": Concise, descriptive, max 60 characters. No quotes around it.
- "description": Engaging, informative, max 254 characters.
- "tags": 5-10 relevant keywords/tags for the image.
- "gps": If you can identify a specific, well-known location in the image, provide GPS coordinates. If unsure or the location is not identifiable, set to null.
- "subject": If you can identify specific known people, bird species, animal species, landmarks, or other notable subjects, list them. If none are identifiable, set to null.

Return ONLY the JSON object, no markdown formatting, no code blocks, no extra text."#
        .to_string()
}

/// Parse the AI response text into an AiResult.
pub fn parse_ai_response(text: &str) -> Result<AiResult> {
    // Try direct JSON parse first
    let cleaned = text.trim();

    // Strip markdown code blocks if present
    let json_str = if cleaned.starts_with("```") {
        let start = cleaned.find('{').unwrap_or(0);
        let end = cleaned.rfind('}').map(|i| i + 1).unwrap_or(cleaned.len());
        &cleaned[start..end]
    } else {
        cleaned
    };

    match serde_json::from_str::<AiResult>(json_str) {
        Ok(result) => Ok(result),
        Err(e) => {
            log::warn!("Failed to parse AI response as JSON: {e}");
            log::debug!("Raw response: {text}");

            // Fallback: try to extract JSON from the response
            if let Some(start) = text.find('{') {
                if let Some(end) = text.rfind('}') {
                    let extracted = &text[start..=end];
                    if let Ok(result) = serde_json::from_str::<AiResult>(extracted) {
                        return Ok(result);
                    }
                }
            }

            anyhow::bail!("Could not parse AI response: {e}")
        }
    }
}
