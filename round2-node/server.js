const express = require('express');
const crypto = require('node:crypto');
const path = require('node:path');
const fs = require('node:fs');
const localEnv = path.join(__dirname, '.env');
if (fs.existsSync(localEnv) && typeof process.loadEnvFile === 'function') process.loadEnvFile(localEnv);
const model = require('./src/modelClient');
const store = require('./src/eventStore');
const { dispatch, testSmtp, sendTest, publicSettings, updateSettings, liveAlertsReady } = require('./src/alertDispatcher');
const { simulate } = require('./src/eventSimulator');

const app = express(); app.use(express.json({ limit: '32kb' })); app.use(express.static(path.join(__dirname, 'public')));
const required = ['log_id', 'timestamp', 'module', 'mode', 'node', 'status', 'delay_ms', 'retry_count', 'message'];
function validate(record) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) return 'body must be one log object';
  for (const field of required) if (!(field in record)) return `missing required field: ${field}`;
  for (const key of ['log_id', 'timestamp', 'module', 'mode', 'node', 'status', 'message']) if (typeof record[key] !== 'string') return `${key} must be a string`;
  if (record.message.length > 8000) return 'message exceeds 8000 characters';
  if (record.reading != null && !Number.isFinite(record.reading)) return 'reading must be a number or null';
  if (!Number.isFinite(record.delay_ms) || !Number.isFinite(record.retry_count)) return 'delay_ms and retry_count must be numeric';
  if (!['Antenna', 'Notice-Bus', 'Lab-Power', 'Life-Support'].includes(record.module)) return 'unsupported module';
  if (!['OK', 'WARN', 'ERROR', 'CRIT'].includes(record.status)) return 'unsupported status';
  if (!['CRUISE', 'DOCKED', 'ALERT', 'MAINT'].includes(record.mode)) return 'unsupported mode';
  return null;
}
app.get('/api/health', async (_req, res) => { let inference; try { inference = await model.health(); } catch (e) { inference = { available: false, error: e.message }; } res.json({ node: 'ok', inference }); });
app.post('/api/predict', async (req, res) => {
  const problem = validate(req.body); if (problem) return res.status(400).json({ error: problem });
  const requestedAt = new Date().toISOString();
  try {
    const prediction = await model.predict(req.body);
    if (prediction.log_id !== req.body.log_id || ![0, 1].includes(prediction.is_true_fault) || !Number.isFinite(prediction.score)) throw new Error('invalid or mismatched inference response');
    const notification = await dispatch(req.body, prediction);
    const event = store.add({ log_id: req.body.log_id, request_id: prediction.request_id, input_timestamp: req.body.timestamp, requested_at: requestedAt, response_at: new Date().toISOString(), model_version: prediction.model_version, is_true_fault: prediction.is_true_fault, score: prediction.score, threshold: prediction.threshold, processing_state: notification.outcome === 'NOT_REQUIRED' ? 'CLASSIFIED' : notification.outcome, notification, simulation_mode: process.env.ALERT_MODE || 'dry-run', inference_ms: prediction.inference_ms });
    res.json({ ...prediction, notification, event_recorded: true });
  } catch (error) { store.add({ log_id: req.body.log_id, request_id: crypto.randomUUID(), input_timestamp: req.body.timestamp, requested_at: requestedAt, processing_state: 'FAILED', failure: error.message, simulation_mode: process.env.ALERT_MODE || 'dry-run' }); res.status(502).json({ error: 'model inference failed', detail: error.message, event_recorded: true }); }
});
app.post('/api/simulate', async (req, res) => { const mode = req.body?.mode || 'dry-run'; if (mode === 'live' && !liveAlertsReady()) return res.status(400).json({ error: 'Live email is not ready. Enable it in Settings and configure a recipient and SMTP password.' }); try { res.json(await simulate(req.body?.limit || 5, mode)); } catch (e) { res.status(500).json({ error: 'simulation failed', detail: e.message }); } });
app.get('/api/events', (_req, res) => res.json(store.recent(Math.min(Number(_req.query.limit) || 100, 500))));
app.get('/api/events/:logId', (req, res) => res.json({ log_id: req.params.logId, history: store.byLogId(req.params.logId) }));
app.get('/api/email/status', async (_req, res) => res.json(await testSmtp()));
app.get('/api/email/settings', (_req, res) => res.json(publicSettings()));
app.post('/api/email/settings', (req, res) => {
  const input = req.body || {};
  const email = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const values = Array.isArray(input.recipient) ? input.recipient : String(input.recipient || '').split(/[;,\r\n]+/);
  const recipients = [...new Set(values.map(value => String(value).trim()).filter(Boolean))];
  if (recipients.length > 10 || recipients.some(value => !email.test(value))) return res.status(400).json({ error: 'Enter up to 10 valid recipient email addresses, separated by commas.' });
  input.recipient = recipients.join(', ');
  try { res.json(updateSettings(input)); }
  catch (error) { res.status(400).json({ error: 'Could not update SMTP settings.' }); }
});
app.post('/api/email/verify', async (_req, res) => { const result = await testSmtp(); res.status(result.verified ? 200 : 400).json(result); });
app.post('/api/email/test', async (_req, res) => { try { const result = await sendTest(); res.status(result.outcome === 'BLOCKED' ? 400 : result.outcome === 'TEST_MESSAGE_ACCEPTED_BY_SMTP' ? 200 : 502).json(result); } catch (e) { res.status(502).json({ outcome: 'SMTP_SEND_FAILED', error: e.message }); } });
const port = Number(process.env.PORT || 3000); app.listen(port, '127.0.0.1', () => console.log(`Rogue Protocol backend listening on http://127.0.0.1:${port}`));
