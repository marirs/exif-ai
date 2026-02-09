use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;

use super::{AiResult, AiService, parse_ai_response};

pub struct GeminiService {
    api_key: String,
    model: String,
    client: Client,
}

impl GeminiService {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl AiService for GeminiService {
    fn name(&self) -> &str {
        "Gemini"
    }

    async fn analyze(&self, image_base64: &str, prompt: &str) -> Result<AiResult> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let body = json!({
            "contents": [
                {
                    "parts": [
                        { "text": prompt },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 1000
            }
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Gemini request failed")?;

        let status = resp.status();
        let text = resp.text().await.context("Failed to read Gemini response")?;

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
