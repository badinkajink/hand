/* MorphoHand control station.
 *
 * Design rules this file follows, each one from a real incident at the bench:
 *
 *  - One poll in flight at a time. The old 750ms setInterval fired regardless of
 *    whether the previous pair of requests had returned, so a slow CB1 turned into a
 *    growing queue of stacked requests and the UI reported failures that were really
 *    just its own backlog.
 *  - A failed poll is a state, not a toast. "Failed to fetch" flashing in the corner
 *    for four seconds and then vanishing is how a dead service looked like a glitch.
 *    Connection loss now latches a banner that stays until it recovers.
 *  - Errors from a command persist. A 409 explaining that the hand is not homed is the
 *    single most useful thing on screen and it used to disappear before it was read.
 *  - Long operations show what they are doing to which axis, with the expected
 *    duration. Homing an axis whose StallGuard2 does not fire looks exactly like a
 *    hang for 46 seconds; the only difference is whether you were told to expect it.
 */

const qs = new URLSearchParams(location.search);
const API = (qs.get('api') || localStorage.getItem('manta-api') || location.origin).replace(/\/$/, '');
const TOKEN = qs.get('token') || localStorage.getItem('manta-token') || '';
localStorage.setItem('manta-api', API);
if (TOKEN) localStorage.setItem('manta-token', TOKEN);
const $ = id => document.getElementById(id);

let state = null, plans = [], eventSeq = 0, polling = false, consecutiveFailures = 0;
let lastError = null, latestRun = null;

const POLL_MS = 750;
const MAX_BACKOFF_MS = 8000;

async function api(path, options = {}) {
  const response = await fetch(`${API}/api/v1${path}`, {
    headers: {'Content-Type': 'application/json', 'X-Manta-Token': TOKEN}, ...options,
  });
  const body = await response.json().catch(() => ({error: `HTTP ${response.status}`}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
const post = (path, body = {}) => api(path, {method: 'POST', body: JSON.stringify(body)});

function toast(message, isError = false) {
  const t = $('toast');
  t.textContent = message;
  t.className = isError ? 'show error' : 'show';
  clearTimeout(t.timer);
  t.timer = setTimeout(() => (t.className = ''), 5000);
}

/* Command errors stay on screen until the next command succeeds. The interlock
 * messages are instructions, not noise. */
function showError(message) {
  lastError = message;
  const box = $('command-error');
  box.textContent = message;
  box.classList.toggle('hidden', !message);
}
function clearError() { showError(null); }

async function action(fn, successMessage) {
  try {
    await fn();
    clearError();
    if (successMessage) toast(successMessage);
    await refresh();
  } catch (e) {
    showError(e.message);
    toast(e.message, true);
  }
}

/* ---------------------------------------------------------------- plans --- */
async function loadPlans() {
  try {
    plans = (await api('/plans')).plans;
    const select = $('plan-select');
    select.innerHTML = '';
    for (const p of plans) {
      const option = document.createElement('option');
      option.value = p.file;
      option.textContent = p.error ? `${p.file} · invalid` : p.design;
      select.appendChild(option);
    }
    showPlan(plans[0]);
  } catch (e) {
    showError(`Plan catalog: ${e.message}`);
  }
}

const unit = (v, u) => (v == null ? null : `${Number(v).toFixed(1)} ${u}`);
const pct = v => (v == null ? null : `${Math.round(Number(v) * 100)}% (sim)`);

function showPlan(p) {
  const meta = p?.meta || {}, metrics = p?.metrics || {}, body = $('metrics-body');
  body.innerHTML = '';
  const rows = [
    ['object', meta.object],
    ['predicted turn', meta.angle_deg != null ? `${meta.angle_deg}°` : null],
    ['careful-bench win', pct(metrics.careful_bench_win)],
    ['full-error win', pct(metrics.full_error_win)],
    ['careful-bench kept', pct(metrics.careful_bench_kept)],
    ['full-error kept', pct(metrics.full_error_kept)],
    ['straddle', unit(meta.straddle_mm, 'mm')],
    ['thumb axial', unit(meta.thumb_axial_mm, 'mm')],
    ['squeeze', unit(meta.squeeze_mm, 'mm')],
    ['grip depth', unit(meta.grip_depth_mm, 'mm')],
    ['pivot axis k', meta.axis_k],
    ['bench height', unit(meta.bench_height_mm, 'mm')],
    ['pad width', unit(meta.pad_width_mm, 'mm')],
    ['fit', p?.violations?.length ? 'OUTSIDE ENVELOPE' : 'Measured envelope'],
  ];
  for (const [k, v] of rows) {
    if (v == null) continue;
    const d = document.createElement('div');
    d.innerHTML = `<small>${k}</small><strong>${v}</strong>`;
    body.appendChild(d);
  }
}

/* ---------------------------------------------------------------- render --- */
function render(s) {
  state = s;
  $('endpoint').textContent = API;

  /* Link banner: a latched serial failure outranks everything else on the page,
   * because nothing else on it is actionable until the board is back. */
  const banner = $('link-banner');
  if (s.link_down) {
    banner.classList.remove('hidden');
    $('link-banner-text').textContent = s.link_down;
  } else {
    banner.classList.add('hidden');
  }

  $('connection').textContent = s.backend === 'real' ? 'Hardware' : `${s.backend} backend`;
  $('health-dot').className = `dot ${s.link_down ? 'offline' : 'online'}`;

  $('session-state').textContent = s.homed ? 'Homed' : 'Not homed';
  $('session-state').className = s.homed ? 'good' : 'warn';
  $('session-detail').textContent = s.homed ? '' : (s.unhomed_reason || 'home required');
  $('operation').textContent = s.operation;
  $('active-design').textContent = s.plan?.design || 'None';
  $('current-pose').textContent = s.current_pose || '—';

  const torque = {1: 'On', 2: 'Off', 3: 'Free'}[s.servo_torque] || '—';
  $('servo-torque').textContent = torque;
  $('servo-torque').className = s.servo_torque === 1 ? 'good' : 'warn';

  const t = s.telemetry || {};
  $('telem-rate').textContent = t.measured_hz ? `${t.measured_hz.toFixed(1)} Hz`
    : (t.error ? 'Failing' : 'Off');
  $('freshness').textContent = t.servo_age_s != null ? `servo ${t.servo_age_s.toFixed(1)} s old`
    : (t.age_s != null ? `${t.age_s.toFixed(1)} s old` : 'No samples');

  $('fit-badge').textContent = s.plan ? (s.plan.violations.length ? 'Does not fit' : 'Plan fits') : 'No plan';
  $('fit-badge').className = `badge ${s.plan?.violations.length ? 'bad' : s.plan ? '' : 'muted'}`;

  $('safety-note').textContent = s.signs_checked
    ? 'Yaw signs supplied for this session. Motion interlock is clear.'
    : 'Yaw direction has not been verified. Finger motion remains interlocked.';
  $('safety-note').className = `callout ${s.signs_checked ? 'good-note' : 'warning'}`;

  const busy = s.busy, blocked = !!s.link_down;
  const motionReady = s.mounts_applied && s.signs_checked && s.servo_torque === 1;
  /* Re-enable whatever a previous disconnect switched off, then apply the interlocks. */
  $('load-plan').disabled = busy;
  /* The M8P keeps its step counters and homing_result across a daemon restart, so a
   * session that starts un-homed against an already-homed board can take the reference
   * over instead of grinding the rails for two minutes. Only offered when it applies. */
  $('adopt-home').classList.toggle('hidden', s.homed || busy || blocked);
  $('adopt-home').disabled = busy || blocked;
  for (const id of ['torque-on', 'torque-off', 'torque-free']) $(id).disabled = busy || blocked;
  $('home').disabled = busy || blocked;
  $('apply-morph').disabled = busy || blocked || !s.homed || !s.plan;
  $('pose-open').disabled = busy || blocked || !motionReady;
  $('pose-grip').disabled = busy || blocked || !motionReady;
  $('run').disabled = busy || blocked || s.current_pose !== 'grip';
  $('stop').disabled = !busy;
  $('disable-motors').disabled = blocked;
  $('reconnect').disabled = !blocked || busy;

  renderProgress(s);
  renderHomeOutcomes(s);
  renderTelemetry(t);
  renderMounts(s.plan?.mounts_palm_mm);
}

/* Live "what is it doing right now", with the expected duration. This is the answer to
 * "is it stuck, or is this normal?" */
function renderProgress(s) {
  const box = $('progress');
  if (!s.busy || !s.home_progress) {
    box.classList.add('hidden');
    return;
  }
  const p = s.home_progress;
  const axis = (s.axes || []).find(a => a.joint === p.joint);
  const label = axis ? `J${p.joint} (${axis.finger} ${axis.axis})` : `J${p.joint}`;
  let text = `${s.operation}: ${label}`;
  if (p.event === 'home_axis_start') {
    text += ` — travelling up to ${axis ? axis.travel_mm.toFixed(0) : '?'} mm, `
      + `up to ${(p.timeout_s || 0).toFixed(0)} s if StallGuard2 does not fire`;
  } else if (p.event === 'home_axis_done') {
    text += ` — ${p.stalled ? 'stalled (StallGuard2)' : 'timeout guarantee'} `
      + `after ${(p.elapsed_s || 0).toFixed(1)} s`;
  } else if (p.event === 'gantry_axis_start') {
    text += ` — moving to ${Number(p.target_mm).toFixed(2)} mm`;
  } else if (p.event === 'home_axis_backoff') {
    text += ' — backing off the hardstop before homing';
  }
  box.textContent = text;
  box.classList.remove('hidden');
}

/* Per-axis home result. "Which axes actually found their hardstop with StallGuard2,
 * and which ground into it for their whole timeout" is a question the operator asked
 * directly, and it was previously only answerable from the CB1's stdout. */
function renderHomeOutcomes(s) {
  const root = $('home-outcomes');
  const outcomes = s.home_outcomes || [];
  if (!outcomes.length) {
    root.innerHTML = '<p class="empty">No home recorded this session.</p>';
    return;
  }
  root.innerHTML = outcomes.map(o => {
    const axis = (s.axes || []).find(a => a.joint === o.joint);
    const how = o.stalled ? 'StallGuard2' : 'timeout';
    return `<div class="axis-row ${o.stalled ? '' : 'by-timeout'}">
        <strong>J${o.joint}</strong>
        <small>${axis ? `${axis.finger} ${axis.axis}` : ''}</small>
        <span class="tag ${o.stalled ? 'good' : 'warn'}">${how}</span>
        <output>${(o.elapsed_s || 0).toFixed(1)} s</output>
      </div>`;
  }).join('');
}

function renderTelemetry(t) {
  const names = ['thumb', 'index', 'middle'], root = $('finger-cards');
  const servos = t.servos, steppers = t.steppers;
  root.innerHTML = '';
  names.forEach((name, i) => {
    const row = document.createElement('div');
    row.className = 'finger-row';
    const v = servos?.[i] || servos?.[String(i)] || {};
    const a = steppers?.[i * 2], b = steppers?.[i * 2 + 1];
    const gantry = a && b
      ? `J${i * 2} ${Number(a.position_mm).toFixed(1)} · J${i * 2 + 1} ${Number(b.position_mm).toFixed(1)} mm`
      : `finger ${i}`;
    const joints = [['aa', 'yaw'], ['fe1', 'mcp'], ['fe2', 'pip']].map(([j, l]) =>
      `<div><small>${l}</small><output>${v[j] != null ? Number(v[j]).toFixed(1) + '°' : '—'}</output></div>`).join('');
    row.innerHTML = `<header><strong>${name}</strong><small>${gantry}</small></header>`
      + `<div class="joint-values">${joints}</div>`;
    root.appendChild(row);
  });

  /* Servo bus health. An SCS chain that has started dropping packets keeps working,
   * slowly and intermittently, long before it fails outright; the retry inside the
   * driver is what hides that, so the count is the only way to see it coming. */
  const bus = t.servo_bus;
  const note = $('telemetry-warning');
  if (t.error) {
    note.textContent = `Telemetry error (${t.consecutive_failures || 1} in a row): ${t.error}`;
    note.className = 'hint bad';
  } else if (bus && bus.timeouts) {
    note.textContent = `Servo bus: ${bus.timeouts} timeouts in ${bus.transactions} transactions `
      + `(${(bus.timeout_rate * 100).toFixed(1)}%), ${bus.consecutive_timeouts} consecutive. `
      + `Retries hide these; a rising rate means the chain is degrading.`;
    note.className = bus.consecutive_timeouts > 2 ? 'hint bad' : 'hint warn';
  } else if (t.servo_polling_suspended) {
    note.textContent = 'Servo polling is suspended while a writer owns the half-duplex bus.';
    note.className = 'hint';
  } else {
    note.textContent = 'Browser refresh reads a cached document and never polls a servo directly.';
    note.className = 'hint';
  }
}

function renderMounts(m) {
  if (!m) return;
  const map = {thumb: 'mount-thumb', index: 'mount-index', middle: 'mount-middle'};
  for (const [f, p] of Object.entries(m)) {
    const g = $(map[f]);
    if (!g) continue;
    const x = 180 + p[0] * 2, y = 115 - p[1] * 1.1;
    const cx = Number(g.querySelector('circle').getAttribute('cx'));
    const cy = Number(g.querySelector('circle').getAttribute('cy'));
    g.setAttribute('transform', `translate(${x - cx} ${y - cy})`);
  }
}

const COMMAND_BUTTONS = ['load-plan', 'home', 'apply-morph', 'pose-open', 'pose-grip',
  'run', 'stop', 'disable-motors', 'torque-on', 'torque-off', 'torque-free'];
function setCommandsEnabled(enabled) {
  for (const id of COMMAND_BUTTONS) $(id).disabled = !enabled;
  $('reconnect').disabled = enabled;
}

/* ----------------------------------------------------------------- poll --- */
async function refresh() {
  if (polling) return;            /* never stack requests on a slow link */
  polling = true;
  try {
    render(await api('/state'));
    const events = (await api(`/events?after=${eventSeq}`)).events;
    if (events.length) {
      eventSeq = events.at(-1).seq;
      for (const e of events) {
        const d = document.createElement('div');
        d.className = `event ${e.level}`;
        d.textContent = `${e.timestamp.slice(11, 19)}  ${e.message}`;
        $('events').prepend(d);
      }
      while ($('events').childElementCount > 300) $('events').lastElementChild.remove();
    }
    consecutiveFailures = 0;
    $('offline-banner').classList.add('hidden');
  } catch (e) {
    consecutiveFailures += 1;
    $('connection').textContent = 'Disconnected';
    $('health-dot').className = 'dot offline';
    /* Nothing can reach the hand, so nothing on the page is actionable. Leaving the
     * buttons live invites a click that produces another "Failed to fetch" and no
     * information; disabling them says plainly that the service is the problem. */
    setCommandsEnabled(false);
    /* Latched, not a toast: a service that has died must not look like a blip. */
    $('offline-banner').classList.remove('hidden');
    $('offline-banner-text').textContent =
      `No response from ${API} (${consecutiveFailures} attempt${consecutiveFailures > 1 ? 's' : ''}): ${e.message}. `
      + `The hand keeps its last commanded position; it is not stopped by this.`;
  } finally {
    polling = false;
  }
}

function scheduleRefresh() {
  const delay = consecutiveFailures
    ? Math.min(MAX_BACKOFF_MS, POLL_MS * 2 ** Math.min(consecutiveFailures, 4))
    : POLL_MS;
  setTimeout(() => refresh().finally(scheduleRefresh), delay);
}

async function refreshLogs() {
  try {
    const logs = (await api('/logs')).logs, root = $('runs');
    root.innerHTML = '';
    if (!logs.length) {
      root.innerHTML = '<p class="empty">No hardware runs recorded.</p>';
      return;
    }
    for (const r of logs.slice(0, 12)) {
      const d = document.createElement('div');
      d.className = 'run';
      const score = r.manual_score
        ? `${r.manual_score.success ? 'success' : 'failure'} · ${r.manual_score.reorientation_deg ?? '—'}°`
        : 'unscored';
      d.innerHTML = `<strong>${r.design} · ${r.status}</strong><small>${r.run_id}<br>${score}</small>`
        + `<div class="actions"><button class="secondary download">Data</button>`
        + `<button class="secondary score">Score</button></div>`;
      d.querySelector('.download').onclick = () => window.open(`${API}/api/v1/logs/${r.run_id}.jsonl`);
      d.querySelector('.score').onclick = () => openScore(r.run_id);
      root.appendChild(d);
    }
  } catch (e) {
    /* Log listing failing is not worth a banner; the poll loop owns connection state. */
  }
}

function openScore(id) { $('score-run-id').value = id; $('score-dialog').showModal(); }

/* ------------------------------------------------------------- controls --- */
$('plan-select').onchange = () => showPlan(plans.find(p => p.file === $('plan-select').value));
$('load-plan').onclick = () => action(
  () => post('/plans/load', {file: $('plan-select').value}), 'Candidate loaded and validated');

$('home').onclick = () => {
  $('home-phrase').value = '';
  const worst = state?.home_worst_case_s;
  $('home-duration').textContent = worst
    ? `Six axes home one at a time. Worst case is about ${Math.round(worst / 60)} minutes `
      + `(${worst.toFixed(0)} s) if StallGuard2 never fires: on this hand J3 and J5 routinely `
      + `do not, and press against their hardstop for their full window (25 s and 24 s). `
      + `That is expected, not a fault.`
    : '';
  $('confirm-home').showModal();
};
$('confirm-home-button').onclick = e => {
  e.preventDefault();
  action(async () => {
    await post('/home', {confirmation: $('home-phrase').value, force: true});
    $('confirm-home').close();
  }, 'Homing started — stay with the hand');
};

$('adopt-home').onclick = () => action(() => post('/home/adopt'),
  "Adopted the board's home — nothing moved");
$('apply-morph').onclick = () => action(() => post('/morphology'), 'Gantry move started');
const motion = () => ({speed_ratio: Number($('speed').value), rate_hz: Number($('rate').value)});
$('pose-open').onclick = () => action(() => post('/pose', {name: 'open', ...motion()}));
$('pose-grip').onclick = () => action(() => post('/pose', {name: 'grip', ...motion()}));
$('run').onclick = () => action(async () => {
  const r = await post('/reorient', motion());
  latestRun = r.run_id;
  toast(`Run ${r.run_id} recording`);
});
$('stop').onclick = () => action(() => post('/stop'), 'Stop sent');
$('disable-motors').onclick = () => {
  if (!confirm('Disable all six steppers and all nine servos?\n\n'
    + 'This drops holding torque everywhere and invalidates the home reference, '
    + 'so the hand must be re-homed afterwards. Use it when an axis is grinding.')) return;
  action(() => post('/motors/disable'), 'Motors disabled — re-home before moving');
};
$('reconnect').onclick = () => action(() => post('/reconnect'), 'Re-checked the link');
$('torque-on').onclick = () => action(() => post('/servos/torque', {state: 'on'}), 'Servo torque ON');
$('torque-off').onclick = () => action(() => post('/servos/torque', {state: 'off'}), 'Servo torque OFF');
$('torque-free').onclick = () => action(() => post('/servos/torque', {state: 'free'}),
  'Servos free — backdrivable by hand');

$('speed').oninput = () => { $('speed-value').textContent = `${Number($('speed').value).toFixed(2)}×`; };
$('refresh-logs').onclick = refreshLogs;
$('save-score').onclick = e => {
  e.preventDefault();
  action(async () => {
    const angle = $('score-angle').value;
    await post(`/logs/${$('score-run-id').value}/score`, {
      success: $('score-success').value === 'true',
      reorientation_deg: angle === '' ? null : Number(angle),
      notes: $('score-notes').value,
    });
    $('score-dialog').close();
    await refreshLogs();
  }, 'Manual score saved');
};
$('dismiss-error').onclick = clearError;

$('endpoint').textContent = API;
loadPlans();
refresh().finally(scheduleRefresh);
refreshLogs();
setInterval(refreshLogs, 5000);
