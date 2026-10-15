const fs = require('node:fs');
const path = require('node:path');

const storePath = path.join(__dirname, '..', 'data', 'events.json');
function readAll() {
  try { return JSON.parse(fs.readFileSync(storePath, 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return []; throw error; }
}
function writeAll(rows) {
  fs.mkdirSync(path.dirname(storePath), { recursive: true });
  const tmp = `${storePath}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(rows, null, 2));
  fs.renameSync(tmp, storePath);
}
function add(event) {
  const rows = readAll();
  rows.push(event);
  writeAll(rows);
  return event;
}
function recent(limit = 100) { return readAll().slice(-limit).reverse(); }
function byLogId(id) { return readAll().filter(row => row.log_id === id); }
function exists(id) { return readAll().some(row => row.log_id === id && row.is_true_fault === 1 && (row.processing_state === 'ALERT_SENT' || row.notification?.outcome === 'ALERT_SENT')); }
module.exports = { add, recent, byLogId, exists };
