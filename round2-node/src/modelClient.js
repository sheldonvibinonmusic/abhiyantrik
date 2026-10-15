const MODEL_URL = process.env.MODEL_URL || 'http://127.0.0.1:8000';

async function request(path, body, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${MODEL_URL}${path}`, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'content-type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`model HTTP ${response.status}: ${payload.detail || payload.error || 'request failed'}`);
    return payload;
  } finally { clearTimeout(timer); }
}

async function health() { return request('/health'); }
async function predict(record) { return request('/predict', record); }
module.exports = { health, predict };
