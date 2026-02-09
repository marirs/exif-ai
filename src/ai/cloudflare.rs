use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;

use super::{AiResult, AiService, parse_ai_response};

pub struct CloudflareService {
    account_id: String,
    api_token: String,
    model: String,
    client: Client,
}

impl CloudflareService {
    pub fn new(account_id: String, api_token: String, model: String) -> Self {
        Self {
            account_id,
            api_token,
            model,
            client: Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl AiService for CloudflareService {
    fn name(&self) -> &str {
        "Cloudflare"
    }

    async fn analyze(&self, image_base64: &str, prompt: &str) -> Result<AiResult> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            self.account_id, self.model
        );

        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "image": image_base64
        });

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&body)
            .send()
            .await
            .context("Cloudflare request failed")?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .context("Failed to read Cloudflare response")?;

        if !status.is_success() {
            anyhow::bail!("Cloudflare API error ({}): {}", status, text);
        }

        let json: serde_json::Value =
            serde_json::from_str(&text).context("Failed to parse Cloudflare response JSON")?;

        let content = json["result"]["response"]
            .as_str()
            .context("No content in Cloudflare response")?;

        parse_ai_response(content)
    }
}
