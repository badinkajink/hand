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
/* ?view=1 is an observer link. Every mutating endpoint is a POST behind the control
 * token and every read is an unauthenticated GET, so a viewer is enforced by the
 * SERVER, not by these disabled buttons -- but a cached token in localStorage would
 * re-arm the page for someone who only meant to watch, so view mode deliberately
 * neither reads nor writes that key. Sharing the URL without ?token= is already
 * read-only for a fresh browser; ?view=1 is what makes it read-only for yours. */
const VIEW_ONLY = qs.get('view') === '1';
const TOKEN = VIEW_ONLY ? '' : (qs.get('token') || localStorage.getItem('manta-token') || '');
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
const post = (path, body = {}) => {
  if (VIEW_ONLY) return Promise.reject(new Error('observer link -- commands are disabled'));
  return api(path, {method: 'POST', body: JSON.stringify(body)});
};

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
      /* The design field is the SEARCH identity, not the plan identity: g12, g12w08
         and g12w11 are three different exports of design g12 and all three read
         "g12" here, which is indistinguishable in the dropdown. The file stem is
         unique by construction (resolve_plan keys on it), so label by that and
         keep the design name only when it adds something. */
      const stem = p.file.replace(/_plan\.json$/, '');
      option.textContent = p.error ? `${p.file} · invalid`
        : (stem === p.design ? stem : `${stem} · ${p.design}`);
      select.appendChild(option);
    }
    showPlan(plans[0]);
  } catch (e) {
    showError(`Plan catalog: ${e.message}`);
  }
}

const unit = (v, u) => (v == null ? null : `${Number(v).toFixed(1)} ${u}`);
const pct = v => (v == null ? null : `${Math.round(Number(v) * 100)}% (sim)`);
/* The clip is the knob that decides whether a plan holds at all -- a design keeps the tool only
   inside a contiguous band of it -- so it belongs on the face of the panel, in both units. */
const clip = v => (v == null ? null : `${Number(v).toFixed(2)} rad = ${(Number(v) * 180 / Math.PI).toFixed(0)}°`);

function showPlan(p) {
  const meta = p?.meta || {}, metrics = p?.metrics || {}, body = $('metrics-body');
  body.innerHTML = '';
  const rows = [
    ['object', meta.object],
    ['predicted turn', meta.angle_deg != null ? `${meta.angle_deg}°` : null],
    ['residual clip', clip(meta.budget_rad)],
    ['clip band', metrics.band_rad ? `${metrics.band_rad[0]}-${metrics.band_rad[1]} rad` : null],
    ['sim held cos', metrics.held_cos == null ? null : Number(metrics.held_cos).toFixed(3)],
    ['finger clearance', unit(metrics.clearance_mm, 'mm')],
    ['expect', metrics.expect],
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
  /* ------------------------------------------------------------------ manual control
 * examples/hand_control.py over this connection. It exists because that script wants
 * the two USB ports for itself, so reaching for it means stopping the service -- and
 * the things you actually want to do by hand (nudge one finger, walk a gantry clear,
 * check a sign) are exactly the things you want to do WITHOUT losing the home and the
 * telemetry.
 *
 * Sliders send on `change`, not `input`: `input` fires on every pixel of a drag and
 * would put a hundred writes on the servo bus for one gesture. */
let manualLimits = null, manualBuilt = false, manualSending = false;

function manualLog(text, kind = '') {
  const box = $('manual-console');
  const line = document.createElement('div');
  line.className = `console-line ${kind}`;
  line.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  box.appendChild(line);
  while (box.children.length > 200) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}

function showPanel(which) {
  $('panel-bench').classList.toggle('hidden', which !== 'bench');
  $('panel-manual').classList.toggle('hidden', which !== 'manual');
  $('view-bench').classList.toggle('active', which === 'bench');
  $('view-manual').classList.toggle('active', which === 'manual');
  localStorage.setItem('manta-view', which);
  if (which === 'manual') loadManualLimits();
}

async function loadManualLimits() {
  if (manualLimits) return manualLimits;
  try {
    manualLimits = await api('/manual/limits');
    buildManualControls();
  } catch (err) {
    manualLog(`could not read joint limits: ${err.message}`, 'bad');
  }
  return manualLimits;
}

function numberRow(id, label, lo, hi, step, value, suffix) {
  return `<div class="knob" data-knob="${id}">
    <div class="knob-head"><small>${label}</small>
      <input id="${id}-num" type="number" min="${lo}" max="${hi}" step="${step}" value="${value}">
      <span>${suffix}</span></div>
    <input id="${id}-range" type="range" min="${lo}" max="${hi}" step="${step}" value="${value}">
    <small class="knob-range">${lo.toFixed(1)} … ${hi.toFixed(1)}</small>
  </div>`;
}

function buildManualControls() {
  if (manualBuilt || !manualLimits) return;
  const L = manualLimits;

  $('manual-mounts').innerHTML = L.fingers.map(f => {
    const mx = L.mounts[f].x, my = L.mounts[f].y;
    const at = (L.mount_positions || {})[f];
    const x = at ? at.x : (mx[0] + mx[1]) / 2, y = at ? at.y : (my[0] + my[1]) / 2;
    return `<div class="manual-finger">
      <header><strong>${f}</strong><small id="mount-${f}-steppers">—</small></header>
      ${numberRow(`mount-${f}-x`, 'palm x', mx[0], mx[1], 0.1, x.toFixed(1), 'mm')}
      ${numberRow(`mount-${f}-y`, 'palm y', my[0], my[1], 0.1, y.toFixed(1), 'mm')}
      <button class="secondary wide" data-mount-go="${f}">Move ${f} gantry</button>
    </div>`;
  }).join('');

  $('manual-joints').innerHTML = L.fingers.map(f => `<div class="manual-finger">
      <header><strong>${f}</strong><small>${L.joints.map(j =>
        `${j}=${L.servo_alias[j]}${L.joint_sign[f][j] < 0 ? '⁻' : ''}`).join('  ')}</small></header>
      ${L.joints.map(j => {
        const [lo, hi] = L.joint_deg[f][j];
        const at = (L.last_command || {})[f]?.[j] ?? 0;
        return numberRow(`joint-${f}-${j}`, j, lo, hi, 0.5, at.toFixed(1), '°');
      }).join('')}
    </div>`).join('');

  /* keep each pair of number box and slider showing the same value */
  for (const el of document.querySelectorAll('[data-knob]')) {
    const id = el.dataset.knob, num = $(`${id}-num`), range = $(`${id}-range`);
    num.oninput = () => { range.value = num.value; };
    range.oninput = () => { num.value = range.value; };
    if (id.startsWith('joint-')) {
      const [, finger, joint] = id.split('-');
      const send = () => sendJoint(finger, joint, Number(num.value));
      range.onchange = send;
      num.onchange = send;
    } else {
      const finger = id.split('-')[1];
      const show = () => showStepperTarget(finger);
      range.onchange = show;
      num.onchange = show;
    }
  }
  for (const btn of document.querySelectorAll('[data-mount-go]')) {
    btn.onclick = () => moveMount(btn.dataset.mountGo);
  }
  for (const f of L.fingers) showStepperTarget(f);
  manualBuilt = true;
}

/* Palm mm and firmware mm are different numbers for the same place, and hand_control.py
 * speaks the other one. Showing both is what keeps them from being confused -- and the
 * conversion is asked of the server, because kinematics is the one place that knows the
 * transform and a copy of it in JavaScript is a copy that will drift. */
const stepperProbe = {};
async function showStepperTarget(finger) {
  const el = $(`mount-${finger}-steppers`);
  if (!el) return;
  const x = Number($(`mount-${finger}-x-num`).value), y = Number($(`mount-${finger}-y-num`).value);
  clearTimeout(stepperProbe[finger]);
  stepperProbe[finger] = setTimeout(async () => {
    try {
      const r = await api('/manual/resolve', {
        method: 'POST',
        body: JSON.stringify({line: `${finger}_x ${x}, ${finger}_y ${y}`}),
      });
      const t = r.mounts[finger];
      el.className = '';
      el.textContent = 'firmware ' + Object.entries(t.steppers)
        .map(([j, mm]) => `J${j} ${mm}`).join('  ');
    } catch (err) {
      el.className = 'bad';
      el.textContent = err.message.replace(/^\w*Error:\s*/, '');
    }
  }, 150);
}

async function sendJoint(finger, joint, deg) {
  if (manualSending) return;
  manualSending = true;
  try {
    await post('/manual/joints', {
      joints: {[finger]: {[joint]: deg}},
      servo_speed: Number($('manual-speed').value),
    });
    manualLog(`${finger}_${joint} → ${deg.toFixed(1)}°`);
    clearError();
  } catch (err) {
    manualLog(`${finger}_${joint} ${deg.toFixed(1)}° refused: ${err.message}`, 'bad');
    showError(err.message);
  } finally {
    manualSending = false;
  }
}

async function moveMount(finger) {
  const x = Number($(`mount-${finger}-x-num`).value);
  const y = Number($(`mount-${finger}-y-num`).value);
  await action(async () => {
    const r = await post('/manual/mounts', {mounts: {[finger]: {x, y}}});
    const t = r.targets[finger];
    manualLog(`${finger} gantry → palm (${x.toFixed(1)}, ${y.toFixed(1)}) mm = firmware `
      + Object.entries(t.steppers).map(([j, mm]) => `J${j} ${mm}`).join(', '));
    manualLimits = null; manualBuilt = false;   // positions changed; re-read on next view
  }, `Moving the ${finger} gantry`);
}

function renderManual(s, {busy, blocked, motionReady}) {
  const mountsOk = !busy && !blocked && s.homed;
  for (const btn of document.querySelectorAll('[data-mount-go]')) btn.disabled = !mountsOk;
  for (const el of document.querySelectorAll('[data-knob]')) {
    const joint = el.dataset.knob.startsWith('joint-');
    const ok = joint ? (!busy && !blocked && motionReady) : mountsOk;
    $(`${el.dataset.knob}-num`).disabled = !ok;
    $(`${el.dataset.knob}-range`).disabled = !ok;
  }
  $('manual-line').disabled = busy || blocked;
  $('manual-send').disabled = busy || blocked;
  $('manual-zero').disabled = busy || blocked || !motionReady;
  $('manual-sync').disabled = busy || blocked;

  const mb = $('manual-mount-badge');
  mb.textContent = s.manual_mounts ? 'hand-placed' : (s.mounts_applied ? 'plan morphology' : 'unknown');
  mb.className = `badge ${s.manual_mounts ? '' : (s.mounts_applied ? '' : 'muted')}`;
  const jb = $('manual-joint-badge');
  jb.textContent = motionReady ? 'live' : 'interlocked';
  jb.className = `badge ${motionReady ? '' : 'muted'}`;
  if (VIEW_ONLY) {
    for (const el of $('panel-manual').querySelectorAll('button, input')) el.disabled = true;
  }
}

function syncManualSliders() {
  if (!manualLimits || !state) return;
  for (const f of manualLimits.fingers) {
    for (const j of manualLimits.joints) {
      const v = state.last_command?.[f]?.[j];
      if (v === undefined) continue;
      const num = $(`joint-${f}-${j}-num`), range = $(`joint-${f}-${j}-range`);
      if (num) { num.value = v.toFixed(1); range.value = v; }
    }
  }
  manualLog('sliders set to the last commanded pose');
}

$('view-bench').onclick = () => showPanel('bench');
$('view-manual').onclick = () => showPanel('manual');
$('manual-speed').oninput = () => { $('manual-speed-value').textContent = $('manual-speed').value; };
$('manual-sync').onclick = syncManualSliders;
$('manual-zero').onclick = () => action(async () => {
  const zero = Object.fromEntries(manualLimits.fingers.map(f =>
    [f, Object.fromEntries(manualLimits.joints.map(j => [j, 0]))]));
  await post('/manual/joints', {joints: zero, servo_speed: Number($('manual-speed').value)});
  manualLog('every joint → 0°');
  syncManualSliders();
}, 'Joints commanded to zero');
$('manual-form').onsubmit = async e => {
  e.preventDefault();
  const line = $('manual-line').value.trim();
  if (!line) return;
  manualLog(`> ${line}`, 'echo');
  try {
    const r = await post('/manual/command', {line, servo_speed: Number($('manual-speed').value)});
    for (const [f, m] of Object.entries(r.mounts || {})) {
      manualLog(`  ${f} gantry → palm (${m.x}, ${m.y}) mm = firmware `
        + Object.entries(m.steppers).map(([j, mm]) => `J${j} ${mm}`).join(', '));
      manualLimits = null; manualBuilt = false;
    }
    if (Object.keys(r.joints || {}).length) {
      manualLog('  ' + Object.entries(r.joints).map(([f, js]) =>
        `${f} ` + Object.entries(js).map(([j, d]) => `${j}=${Number(d).toFixed(1)}`).join(' ')
      ).join('   '));
    }
    $('manual-line').value = '';
    clearError();
  } catch (err) {
    /* An error here belongs in the console next to the line that caused it, not only in
     * the page-wide banner -- the whole point of a console is that it keeps the pair. */
    manualLog(`  ${err.message}`, 'bad');
    showError(err.message);
  }
  await refresh();
};
if (localStorage.getItem('manta-view') === 'manual') showPanel('manual');

$('endpoint').textContent = VIEW_ONLY ? `${API}  (observer)` : API;

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
  /* A manual gantry move retires the plan's morphology but leaves the rails at a known,
   * bounds-checked place -- which is all the finger interlock actually needs. */
  const motionReady = (s.mounts_applied || s.manual_mounts) && s.signs_checked
    && s.servo_torque === 1;
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
  for (const id of ['pose-open', 'setup-open', 'setup-grip'])
    $(id).disabled = busy || blocked || !motionReady;   /* same interlock as the run tab */
  $('pose-grip').disabled = busy || blocked || !motionReady;
  $('run').disabled = busy || blocked || s.current_pose !== 'grip';
  $('stop').disabled = !busy;
  $('disable-motors').disabled = blocked;
  $('reconnect').disabled = !blocked || busy;
  if (VIEW_ONLY) setCommandsEnabled(false);   // render() re-derives every button from
  // the runtime state, so the mode has to be re-asserted after it, not just at boot.

  renderProgress(s);
  renderHomeOutcomes(s);
  renderTracker(s.tracker, s.tracker_service);
  renderTelemetry(t);
  renderMounts(s.plan?.mounts_palm_mm);
  renderManual(s, {busy, blocked, motionReady});
}

/* Where the shaft actually is. Before 2026-08-31 the only answer to that on this station
 * was the operator's eye, so the guiding rule here is that a number is shown only when it
 * was measured: a stale sample says so and keeps its last value greyed, a lost tag says
 * LOST rather than freezing, and bench x/y is blank until the heading is calibrated. */
const degFromUp = c => (Math.acos(Math.min(1, Math.max(-1, c))) * 180 / Math.PI);

function renderTracker(tr, service) {
  const badge = $('track-badge'), body = $('track-body'), empty = $('track-empty');
  const serviceNote = $('track-service-note');
  if (serviceNote) {
    serviceNote.textContent = service
      ? (service.error
        ? `Automatic tracker configured, but its last arm failed: ${service.error}`
        : 'Automatic tracker ready: Run reorientation arms it before motion and finalizes it afterward.')
      : 'Automatic tracking is not configured on the CB1. Configure the workstation tracker service before a production run.';
  }
  if (!tr || !tr.received) {
    badge.textContent = service ? (service.error ? 'tracker fault' : 'automatic') : 'No tracker';
    badge.className = `badge ${service?.error ? 'bad' : 'muted'}`;
    body.classList.add('hidden'); empty.classList.remove('hidden');
    return;
  }
  body.classList.remove('hidden'); empty.classList.add('hidden');
  const last = tr.last || {}, seen = last.seen !== false, cos = last.cos;
  if (!tr.fresh) {
    badge.textContent = `stale ${tr.age_s == null ? '' : tr.age_s.toFixed(0) + ' s'}`;
    badge.className = 'badge bad';
  } else if (!seen) {
    badge.textContent = 'tag LOST'; badge.className = 'badge bad';
  } else {
    badge.textContent = `tracking · ${tr.received}`; badge.className = 'badge';
  }
  const el = $('track-cos');
  el.textContent = cos == null ? '—'
    : `${cos >= 0 ? '+' : ''}${Number(cos).toFixed(3)}   ${degFromUp(cos).toFixed(1)}° from up`;
  el.style.opacity = (tr.fresh && seen) ? 1 : .45;
  /* The bar runs from the centre so the direction of the turn is visible, not just its
     size; red on the left half is the wrong pole. */
  const fill = $('track-fill'), c = cos == null ? 0 : Math.min(1, Math.max(-1, cos));
  fill.style.left = `${50 + Math.min(0, c) * 50}%`;
  fill.style.width = `${Math.abs(c) * 50}%`;
  fill.className = c < 0 ? 'bad' : '';

  const turned = (tr.start_cos != null && cos != null)
    ? degFromUp(tr.start_cos) - degFromUp(cos) : null;
  const fall = (tr.start_z_mm != null && tr.min_z_mm != null)
    ? tr.start_z_mm - tr.min_z_mm : null;
  const rows = [
    ['turned so far', turned == null ? null
      : `${turned >= 0 ? '+' : ''}${turned.toFixed(1)}° toward vertical`],
    ['peak cos', tr.peak_cos == null ? null : Number(tr.peak_cos).toFixed(3)],
    ['lowest cos', tr.min_cos == null ? null
      : Number(tr.min_cos).toFixed(3) + (tr.min_cos < -0.15 ? ' — WRONG POLE' : '')],
    ['height, bench floor', last.z_bench_mm == null ? null
      : `${Number(last.z_bench_mm).toFixed(0)} mm`],
    ['height, simulator z', last.z_sim_mm == null ? null
      : `${Number(last.z_sim_mm).toFixed(0)} mm (bench scene stands it at 100)`],
    ['fallen from start', fall == null ? null : `${fall.toFixed(0)} mm`],
    ['bench x / y', (last.x_bench_mm === '' || last.x_bench_mm == null) ? 'no heading calibrated'
      : `${Number(last.x_bench_mm).toFixed(0)} / ${Number(last.y_bench_mm).toFixed(0)} mm`],
    ['run', tr.run_id || 'not tied to a run'],
  ];
  const grid = $('track-grid');
  grid.innerHTML = '';
  for (const [k, v] of rows) {
    if (v == null) continue;
    const d = document.createElement('div');
    d.innerHTML = `<small>${k}</small><strong>${v}</strong>`;
    grid.appendChild(d);
  }
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
  const on = enabled && !VIEW_ONLY;
  for (const id of COMMAND_BUTTONS) $(id).disabled = !on;
  $('adopt-home').disabled = !on;
  $('reconnect').disabled = VIEW_ONLY ? true : enabled;
}

function enterViewOnlyMode() {
  const banner = document.createElement('div');
  banner.className = 'banner progress';
  banner.innerHTML = '<strong>Observer link.</strong> <span>Live telemetry, no control. ' +
    'Every command button is disabled and this page holds no control token. ' +
    'Drop <code>?view=1</code> from the URL and add <code>?token=&hellip;</code> to take over.</span>';
  document.querySelector('main').prepend(banner);
  setCommandsEnabled(false);
  for (const el of document.querySelectorAll('.tab')) el.disabled = true;
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
      /* Measured and estimated are shown side by side rather than one replacing the
         other: they are two readings of the same run and their disagreement is data. */
      const tk = r.object_track;
      const measured = tk && tk.seen
        ? `<br>tag: ${tk.deg_turned == null ? '—' : (tk.deg_turned >= 0 ? '+' : '')
            + tk.deg_turned.toFixed(0) + '°'}`
          + `, peak cos ${tk.cos_peak == null ? '—' : tk.cos_peak.toFixed(2)}`
          + (tk.dropped ? ' · DROPPED' : '') + (tk.wrong_pole ? ' · wrong pole' : '')
        : '';
      d.innerHTML = `<strong>${r.design} · ${r.status}</strong><small>${r.run_id}<br>${score}${measured}</small>`
        + `<div class="actions"><button class="secondary download">Data</button>`
        + `<button class="secondary score">Score</button></div>`;
      d.querySelector('.download').onclick = () => window.open(`${API}/api/v1/logs/${r.run_id}.jsonl`);
      d.querySelector('.score').onclick = () => openScore(r.run_id, r.object_track);
      root.appendChild(d);
    }
  } catch (e) {
    /* Log listing failing is not worth a banner; the poll loop owns connection state. */
  }
}

function openScore(id, track) {
  $('score-run-id').value = id;
  /* Seeded from the tag when there is one, so the operator confirms a measurement instead
     of recalling an angle. Left editable: the instrument can be wrong too (a dropout
     during the turn, a slipped shaft) and the operator saw the run. */
  const hint = $('score-measured');
  if (track && track.seen && track.deg_turned != null) {
    $('score-angle').value = Math.round(track.deg_turned);
    hint.textContent = `AprilTag measured ${track.deg_turned.toFixed(1)}° `
      + `(peak cos ${track.cos_peak?.toFixed(3) ?? '—'}, `
      + `${Math.round((track.visibility ?? 0) * 100)}% visible`
      + (track.dropped ? ', DROPPED' : '') + ').';
    hint.classList.remove('hidden');
    $('score-success').value = String(!track.dropped);
  } else {
    $('score-angle').value = '';
    hint.textContent = 'No tag trace for this run — this is an eyeball estimate.';
    hint.classList.remove('hidden');
  }
  $('score-dialog').showModal();
}

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
// Setup-card duplicates of the two keyframe moves.  They are the same server call
// as the run-tab buttons, but they belong here too: the grasp keyframe is what you
// RESET to between attempts, and having to leave the setup card to find it is how a
// repeat ends up starting from the previous attempt's end pose.
$('setup-open').onclick = () => action(() => post('/pose', {name: 'open', ...motion()}),
                                       'Moving to the open keyframe');
$('setup-grip').onclick = () => action(() => post('/pose', {name: 'grip', ...motion()}),
                                       'Moving to the grasp keyframe');
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

$('endpoint').textContent = VIEW_ONLY ? `${API}  (observer)` : API;
if (VIEW_ONLY) enterViewOnlyMode();
loadPlans();
refresh().finally(scheduleRefresh);
refreshLogs();
setInterval(refreshLogs, 5000);
