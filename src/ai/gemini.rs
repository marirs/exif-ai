use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

use super::{AiResult, AiService, parse_ai_response};

/// Maximum time to wait for a single AI request before giving up.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

pub struct GeminiService {
    api_key: String,
    model: String,
    client: Client,
}

impl GeminiService {
    pub fn new(api_key: String, model: String) -> Self {
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            api_key,
            model,
            client,
        }
    }
}

#[async_trait::async_trait]
impl AiService for GeminiService {
    fn name(&self) -> &str {
        "Gemini"
    }

    async fn analyze(&self, image_base64: &str, prompt: &str, mime_type: &str) -> Result<AiResult> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.model
        );

        let body = json!({
            "contents": [
                {
                    "parts": [
                        { "text": prompt },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 2000
            }
        });

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("Gemini request failed")?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .context("Failed to read Gemini response")?;

        if !status.is_success() {
            anyhow::bail!("Gemini API error ({}): {}", status, text);
        }

        let json: serde_json::Value =
            serde_json::from_str(&text).context("Failed to parse Gemini response JSON")?;

        let content = json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .context("No content in Gemini response")?;

        parse_ai_response(content)
    }
}
