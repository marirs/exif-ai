use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;

use super::{AiResult, AiService, parse_ai_response};

pub struct OpenAiService {
    api_key: String,
    model: String,
    client: Client,
}

impl OpenAiService {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl AiService for OpenAiService {
    fn name(&self) -> &str {
        "OpenAI"
    }

    async fn analyze(&self, image_base64: &str, prompt: &str) -> Result<AiResult> {
        let body = json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an image analysis assistant. You MUST respond with valid JSON only. No markdown, no code blocks, no extra text. All string values MUST be enclosed in double quotes."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:image/jpeg;base64,{image_base64}"),
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "response_format": { "type": "json_object" }
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .context("OpenAI request failed")?;

        let status = resp.status();
        let text = resp.text().await.context("Failed to read OpenAI response")?;

        if !status.is_success() {
            anyhow::bail!("OpenAI API error ({}): {}", status, text);
        }

        let json: serde_json::Value =
            serde_json::from_str(&text).context("Failed to parse OpenAI response JSON")?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .context("No content in OpenAI response")?;

        parse_ai_response(content)
    }
}
