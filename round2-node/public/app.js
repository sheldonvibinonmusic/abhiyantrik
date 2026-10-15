const $ = id => document.getElementById(id);

async function loadStatus() {
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    const settings = await (await fetch('/api/email/settings')).json();
    $('backend').textContent = health.node === 'ok' ? 'Online' : 'Unavailable';
    $('model').textContent = health.inference?.available ? `Available · ${health.inference.model_version || ''}` : 'Unavailable';
    $('emailMode').textContent = settings.liveAlerts ? 'Live alerts enabled' : 'Dry run · no emails';
    document.querySelector('.routing-note').innerHTML = settings.recipient
      ? `<strong>Email recipient:</strong> ${settings.recipient}. ${settings.liveAlerts ? '<strong>Live fault emails enabled.</strong>' : '<strong>Dry run; no email is sent.</strong>'}`
      : '<strong>Email routing:</strong> No recipient is configured. Fault alerts are simulated locally and <strong>no email is sent</strong>.';
  } catch (error) {
    $('backend').textContent = `Unavailable · ${error.message}`;
    $('model').textContent = 'Unknown';
  }
}

async function loadEvents() {
  const rows = await (await fetch('/api/events')).json();
  const completed = rows.filter(event => event.is_true_fault === 0 || event.is_true_fault === 1);
  $('events').innerHTML = completed.length ? completed.map(event => {
    const outcome = event.notification?.outcome;
    const label = outcome === 'ALERT_SENT' ? `SMTP ACCEPTED · ${event.notification.recipient}`
      : outcome === 'DUPLICATE_SKIPPED' ? 'DUPLICATE SKIPPED'
        : outcome === 'SIMULATION_EMAIL_LIMIT' ? 'EMAIL CAP · NOT SENT'
          : outcome === 'SMTP_SEND_FAILED' ? 'SMTP SEND FAILED'
            : outcome === 'ALERT_SIMULATED' ? 'DRY RUN · NOT SENT'
              : event.is_true_fault ? 'ALERT NOT SENT' : 'NOT REQUIRED';
    return `<tr><td>${event.log_id}</td><td>${event.is_true_fault ? 'FAULT' : 'NO FAULT'}</td><td>${Number(event.score).toFixed(4)}</td><td>${event.processing_state}</td><td>${label}</td></tr>`;
  }).join('') : '<tr><td colspan="5" class="empty-state">No recent activities. Submit a log or start a simulation to see results here.</td></tr>';
}

$('predictForm').addEventListener('submit', async event => {
  event.preventDefault();
  const body = Object.fromEntries(new FormData(event.target));
  body.reading = body.reading === '' ? null : Number(body.reading);
  body.delay_ms = Number(body.delay_ms);
  body.retry_count = Number(body.retry_count);
  $('predictionResult').textContent = 'Request in progress…';
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    const result = await response.json();
    $('predictionResult').textContent = response.ok ? JSON.stringify(result, null, 2) : `Request failed (HTTP ${response.status})\n${JSON.stringify(result, null, 2)}`;
    await loadEvents();
  } catch (error) { $('predictionResult').textContent = `Request failed: ${error.message}`; }
});

$('simulate').addEventListener('click', async () => {
  const mode = $('simMode').value;
  if (mode === 'live' && !confirm('Run a random sample and allow up to one real fault email to the configured recipient? Predicted non-faults will not send email.')) return;
  $('simulationResult').textContent = 'Choosing random records and running predictions…';
  try {
    const response = await fetch('/api/simulate', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ limit: Number($('simLimit').value), mode }) });
    const result = await response.json();
    $('simulationResult').textContent = JSON.stringify(result, null, 2);
    await loadEvents();
  } catch (error) { $('simulationResult').textContent = `Simulation failed: ${error.message}`; }
});

$('refresh').addEventListener('click', () => { loadStatus(); loadEvents(); });
const dialog = $('settingsDialog');
function formSettings() {
  const form = new FormData($('settingsForm'));
  return { recipient: form.get('recipient'), liveAlerts: form.get('liveAlerts') === 'on' };
}
async function loadEmailSettings() {
  const settings = await (await fetch('/api/email/settings')).json();
  const form = $('settingsForm');
  form.elements.recipient.value = settings.recipient || '';
  form.elements.liveAlerts.checked = settings.liveAlerts;
  $('smtpHost').textContent = settings.host || 'not configured';
  $('smtpPort').textContent = `${settings.port || '—'}${settings.secure ? ' · implicit TLS' : ' · STARTTLS'}`;
  $('smtpSender').textContent = settings.from || 'not configured';
  $('settingsResult').textContent = settings.smtpCredentialConfigured
    ? 'SMTP App Password is loaded from .env.'
    : 'SMTP App Password is missing from .env or Node needs a restart to load it.';
}
async function saveEmailSettings() {
  const response = await fetch('/api/email/settings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(formSettings()) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Invalid settings');
  await loadStatus();
  return result;
}
$('openSettings').addEventListener('click', async () => { await loadEmailSettings(); dialog.showModal(); });
$('closeSettings').addEventListener('click', () => dialog.close());
$('settingsForm').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const result = await saveEmailSettings();
    $('settingsResult').textContent = `Email settings saved. SMTP credential loaded from .env: ${result.smtpCredentialConfigured ? 'yes' : 'no'}.`;
  } catch (error) { $('settingsResult').textContent = `Could not save settings: ${error.message}`; }
});
$('verifySmtp').addEventListener('click', async () => {
  $('settingsResult').textContent = 'Verifying SMTP connection…';
  try {
    await saveEmailSettings();
    const response = await fetch('/api/email/verify', { method: 'POST' });
    const result = await response.json();
    $('settingsResult').textContent = JSON.stringify(result, null, 2);
  } catch (error) { $('settingsResult').textContent = `SMTP verification failed: ${error.message}`; }
});
$('sendTestEmail').addEventListener('click', async () => {
  if (!confirm('Send one test email to the recipient shown in Settings?')) return;
  $('settingsResult').textContent = 'Sending one test email…';
  try {
    await saveEmailSettings();
    const response = await fetch('/api/email/test', { method: 'POST' });
    const result = await response.json();
    $('settingsResult').textContent = JSON.stringify(result, null, 2);
  } catch (error) { $('settingsResult').textContent = `Test email failed: ${error.message}`; }
});

loadStatus();
loadEvents();
