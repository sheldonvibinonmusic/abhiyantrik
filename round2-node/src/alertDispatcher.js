const nodemailer = require('nodemailer');

const settings = {
  host: process.env.SMTP_HOST || 'smtp.gmail.com',
  port: Number(process.env.SMTP_PORT || 587),
  secure: process.env.SMTP_SECURE === 'true',
  user: process.env.SMTP_USER || process.env.SMTP_FROM || '4714us@gmail.com',
  pass: process.env.SMTP_PASS || '',
  from: process.env.SMTP_FROM || '4714us@gmail.com',
  recipient: process.env.SMTP_TEST_RECIPIENT || '',
  liveAlerts: process.env.ALERT_MODE === 'live'
};

function publicSettings() {
  return { host: settings.host, port: settings.port, secure: settings.secure, user: settings.user,
    from: settings.from, recipient: settings.recipient, liveAlerts: settings.liveAlerts,
    smtpCredentialConfigured: Boolean(settings.pass) };
}
function updateSettings(input) {
  settings.recipient = input.recipient;
  settings.liveAlerts = Boolean(input.liveAlerts);
  return publicSettings();
}
function recipientList() { return String(settings.recipient || '').split(/[;,\r\n]+/).map(value => value.trim()).filter(Boolean); }
function makeTransport() {
  if (!settings.host || !settings.port || !settings.user || !settings.pass) return null;
  return nodemailer.createTransport({ host: settings.host, port: settings.port, secure: settings.secure,
    auth: { user: settings.user, pass: settings.pass }, connectionTimeout: 8000,
    greetingTimeout: 8000, socketTimeout: 12000 });
}
async function testSmtp() {
  const transport = makeTransport();
  if (!transport) return { configured: false, verified: false, outcome: 'SMTP_SETTINGS_INCOMPLETE' };
  try { await transport.verify(); return { configured: true, verified: true, outcome: 'SMTP_CONFIGURATION_VERIFIED' }; }
  catch (error) { return { configured: true, verified: false, outcome: 'SMTP_VERIFY_FAILED', error: error.message }; }
}
async function dispatch(record, prediction, options = {}) {
  if (prediction.is_true_fault !== 1) return { outcome: 'NOT_REQUIRED' };
  const live = options.mode ? options.mode === 'live' : settings.liveAlerts;
  if (!live) return { outcome: 'ALERT_SIMULATED', recipient: null, detail: 'Dry run; no email was sent.' };
  const recipients = recipientList();
  if (!recipients.length) return { outcome: 'ROUTING_NOT_CONFIGURED', detail: 'Configure an alert recipient in Settings.' };
  const transport = makeTransport();
  if (!transport) return { outcome: 'SMTP_SETTINGS_INCOMPLETE' };
  await transport.verify();
  const result = await transport.sendMail({ from: settings.from, to: recipients,
    subject: `[Rogue Protocol] Fault predicted: ${record.module} (${record.log_id})`,
    text: `Event: ${record.log_id}\nTimestamp: ${record.timestamp}\nModule: ${record.module}\nStatus: ${record.status}\nMessage: ${record.message}\nReading: ${record.reading ?? 'not provided'}\nModel decision: FAULT (${prediction.score.toFixed(4)}; threshold ${prediction.threshold})\n\nReview the event in the Rogue Protocol incident console.` });
  return { outcome: 'ALERT_SENT', smtp_outcome: 'ACCEPTED_BY_SMTP', recipient: recipients, messageId: result.messageId, accepted: result.accepted };
}
// Explicit simulation requests may send when SMTP and recipients are configured.
function liveAlertsReady() { return recipientList().length > 0 && Boolean(makeTransport()); }
async function sendTest() {
  const recipients = recipientList();
  if (!recipients.length) return { outcome: 'BLOCKED', detail: 'Set at least one recipient address in Settings first.' };
  const status = await testSmtp();
  if (!status.verified) return status;
  const result = await makeTransport().sendMail({ from: settings.from, to: recipients,
    subject: 'Rogue Protocol SMTP integration test', text: 'Controlled test message. No incident is associated with this email.' });
  return { outcome: 'TEST_MESSAGE_ACCEPTED_BY_SMTP', messageId: result.messageId, accepted: result.accepted, recipient: recipients };
}
module.exports = { publicSettings, updateSettings, testSmtp, dispatch, sendTest, liveAlertsReady };
