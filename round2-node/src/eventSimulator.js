const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const { predict } = require('./modelClient');
const store = require('./eventStore');
const { dispatch, liveAlertsReady } = require('./alertDispatcher');

function csvRow(line) {
  const cols = []; let cur = ''; let quoted = false;
  for (let i = 0; i < line.length; i++) { const c = line[i]; if (c === '"' && line[i + 1] === '"' && quoted) { cur += '"'; i++; } else if (c === '"') quoted = !quoted; else if (c === ',' && !quoted) { cols.push(cur); cur = ''; } else cur += c; }
  cols.push(cur); return cols;
}
function parseValue(key, value) { if (value === '') return null; if (['reading', 'delay_ms', 'retry_count', 'is_true_fault'].includes(key)) return Number(value); return value; }

async function randomSample(limit) {
  const file = path.resolve(__dirname, '..', '..', 'test.csv');
  const rl = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
  let headers; const all = [];
  for await (const line of rl) {
    if (!headers) { headers = csvRow(line); continue; }
    const cells = csvRow(line);
    all.push(Object.fromEntries(headers.map((key, i) => [key, parseValue(key, cells[i] || '')])));
  }
  for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
  return all.slice(0, limit);
}

async function simulate(limit = 5, mode = 'dry-run') {
  const bounded = Math.min(Math.max(Number(limit) || 1, 1), Number(process.env.SIMULATION_MAX_EVENTS || 10), 3000);
  if (!['dry-run', 'live'].includes(mode)) throw new Error('mode must be dry-run or live');
  if (mode === 'live' && !liveAlertsReady()) throw new Error('Live simulation needs enabled live alerts, a recipient, and SMTP credentials.');
  const records = await randomSample(bounded);
  const maxEmails = mode === 'live' ? Math.max(0, Math.min(3, Number(process.env.MAX_SIMULATION_ALERT_EMAILS || 1))) : 0;
  const startedAt = Date.now(); let emailSlotsUsed = 0;
  const result = { mode, total_events_processed: 0, prediction_responses_received: 0, faults_predicted: 0, failed_model_requests: 0, failed_notification_requests: 0, emails_simulated: 0, emails_sent: 0, duplicates_skipped: 0, runtime_ms: 0, events: [] };
  const width = Math.min(4, Math.max(1, Number(process.env.SIMULATION_BATCH_SIZE || 4)));
  for (let i = 0; i < records.length; i += width) {
    const settled = await Promise.all(records.slice(i, i + width).map(async record => {
      const requestedAt = new Date().toISOString();
      let prediction;
      try {
        prediction = await predict(record);
        if (prediction.log_id !== record.log_id || ![0, 1].includes(prediction.is_true_fault) || !Number.isFinite(prediction.score)) throw new Error('invalid or mismatched model response');
      } catch (error) {
        result.failed_model_requests++;
        return store.add({ log_id: record.log_id, request_id: null, requested_at: requestedAt, response_at: null, processing_state: 'FAILED', failure: error.message, simulation_mode: mode });
      }
      result.prediction_responses_received++; result.total_events_processed++;
      if (prediction.is_true_fault) result.faults_predicted++;
      let notification;
      if (store.exists(record.log_id)) {
        notification = { outcome: 'DUPLICATE_SKIPPED', detail: 'An earlier alert for this event was accepted by SMTP.' };
        result.duplicates_skipped++;
      } else if (mode === 'live' && prediction.is_true_fault === 1 && emailSlotsUsed >= maxEmails) {
        notification = { outcome: 'SIMULATION_EMAIL_LIMIT', detail: `Per-run live email cap is ${maxEmails}.` };
      } else {
        if (mode === 'live' && prediction.is_true_fault === 1) emailSlotsUsed++;
        try {
          notification = await dispatch(record, prediction, { mode });
          if (notification.outcome === 'ALERT_SIMULATED') result.emails_simulated++;
          if (notification.outcome === 'ALERT_SENT') result.emails_sent++;
        } catch (error) {
          notification = { outcome: 'SMTP_SEND_FAILED', detail: error.message };
          result.failed_notification_requests++;
        }
      }
      const state = notification.outcome === 'NOT_REQUIRED' ? 'CLASSIFIED' : notification.outcome;
      return store.add({ log_id: record.log_id, request_id: prediction.request_id, input_timestamp: record.timestamp, requested_at: requestedAt, response_at: new Date().toISOString(), model_version: prediction.model_version, is_true_fault: prediction.is_true_fault, score: prediction.score, threshold: prediction.threshold, processing_state: state, notification, simulation_mode: mode, inference_ms: prediction.inference_ms });
    }));
    result.events.push(...settled);
  }
  result.runtime_ms = Date.now() - startedAt;
  return result;
}
module.exports = { simulate };
