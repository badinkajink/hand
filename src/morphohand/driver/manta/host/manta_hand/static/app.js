const qs = new URLSearchParams(location.search);
const API = (qs.get('api') || localStorage.getItem('manta-api') || location.origin).replace(/\/$/, '');
const TOKEN = qs.get('token') || localStorage.getItem('manta-token') || '';
localStorage.setItem('manta-api', API);
if (TOKEN) localStorage.setItem('manta-token', TOKEN);
const $ = id => document.getElementById(id);
let state = null, plans = [], eventSeq = 0, latestRun = null;

async function api(path, options={}) {
  const response = await fetch(`${API}/api/v1${path}`, {headers:{'Content-Type':'application/json','X-Manta-Token':TOKEN}, ...options});
  const body = await response.json().catch(()=>({error:`HTTP ${response.status}`}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
const post = (path, body={}) => api(path, {method:'POST', body:JSON.stringify(body)});
function toast(message, error=false){const t=$('toast');t.textContent=message;t.className=error?'show error':'show';clearTimeout(t.timer);t.timer=setTimeout(()=>t.className='',4000)}
async function action(fn){try{await fn();await refresh()}catch(e){toast(e.message,true)}}

async function loadPlans(){
  try{plans=(await api('/plans')).plans;const s=$('plan-select');s.innerHTML='';
    for(const p of plans){const o=document.createElement('option');o.value=p.file;o.textContent=p.error?`${p.file} · invalid`:p.design;s.appendChild(o)}
    showPlan(plans[0]);
  }catch(e){toast(`Plan catalog: ${e.message}`,true)}
}
function showPlan(p){
  const meta=p?.meta||{}, metrics=p?.metrics||{}, body=$('metrics-body'); body.innerHTML='';
  const keys=[['object',meta.object],['predicted turn',meta.angle_deg!=null?`${meta.angle_deg}°`:null],['careful-bench win',pct(metrics.careful_bench_win)],['full-error win',pct(metrics.full_error_win)],['careful-bench kept',pct(metrics.careful_bench_kept)],['full-error kept',pct(metrics.full_error_kept)],['straddle',unit(meta.straddle_mm,'mm')],['thumb axial',unit(meta.thumb_axial_mm,'mm')],['squeeze',unit(meta.squeeze_mm,'mm')],['grip depth',unit(meta.grip_depth_mm,'mm')],['pivot axis k',meta.axis_k],['bench height',unit(meta.bench_height_mm,'mm')],['pad width',unit(meta.pad_width_mm,'mm')],['fit',p?.violations?.length?'OUTSIDE ENVELOPE':'Measured envelope']];
  for(const [k,v] of keys){if(v==null)continue;const d=document.createElement('div');d.innerHTML=`<small>${k}</small><strong>${v}</strong>`;body.appendChild(d)}
}
const unit=(v,u)=>v==null?null:`${Number(v).toFixed(1)} ${u}`;
const pct=v=>v==null?null:`${Math.round(Number(v)*100)}% (sim)`;

function render(s){
  state=s;$('connection').textContent=`${s.backend} backend`;$('endpoint').textContent=API;$('health-dot').className='dot online';
  $('session-state').textContent=s.homed?'Homed':'Not homed';$('operation').textContent=s.operation;$('active-design').textContent=s.plan?.design||'None';$('current-pose').textContent=s.current_pose||'—';
  $('fit-badge').textContent=s.plan?(s.plan.violations.length?'Does not fit':'Plan fits'):'No plan';$('fit-badge').className=`badge ${s.plan?.violations.length?'bad':s.plan?'':'muted'}`;
  const t=s.telemetry;$('telem-rate').textContent=t.measured_hz?`${t.measured_hz.toFixed(1)} Hz`:(t.error?'Unavailable':'Off');$('freshness').textContent=t.servo_age_s!=null?`servo ${t.servo_age_s.toFixed(1)} s old`:(t.age_s!=null?`${t.age_s.toFixed(1)} s old`:'No samples');
  $('safety-note').textContent=s.signs_checked?'Yaw signs supplied for this session. Motion interlock is clear.':'Yaw direction has not been verified. Finger motion remains interlocked.';$('safety-note').className=`callout ${s.signs_checked?'good-note':'warning'}`;
  const busy=s.busy; $('home').disabled=busy;$('apply-morph').disabled=busy||!s.homed||!s.plan;$('pose-open').disabled=busy||!s.mounts_applied||!s.signs_checked;$('pose-grip').disabled=busy||!s.mounts_applied||!s.signs_checked;$('run').disabled=busy||s.current_pose!=='grip';
  $('stop').disabled=!busy;
  renderTelemetry(t.servos,t.steppers);renderMounts(s.plan?.mounts_palm_mm);if(t.error)$('telemetry-warning').textContent=t.error;
}
function renderTelemetry(servos,steppers){
  const names=['thumb','index','middle'], root=$('finger-cards');root.innerHTML='';
  names.forEach((name,i)=>{const row=document.createElement('div');row.className='finger-row';const v=servos?.[i]||servos?.[String(i)]||{},a=steppers?.[i*2],b=steppers?.[i*2+1],gantry=a&&b?`J${i*2} ${Number(a.position_mm).toFixed(1)} · J${i*2+1} ${Number(b.position_mm).toFixed(1)} mm`:`finger ${i}`;row.innerHTML=`<header><strong>${name}</strong><small>${gantry}</small></header><div class="joint-values">${[['aa','yaw'],['fe1','mcp'],['fe2','pip']].map(([j,l])=>`<div><small>${l}</small><output>${v[j]!=null?Number(v[j]).toFixed(1)+'°':'—'}</output></div>`).join('')}</div>`;root.appendChild(row)})
}
function renderMounts(m){if(!m)return;const map={thumb:'mount-thumb',index:'mount-index',middle:'mount-middle'};for(const [f,p] of Object.entries(m)){const g=$(map[f]),x=180+p[0]*2,y=115-p[1]*1.1;g.setAttribute('transform',`translate(${x-Number(g.querySelector('circle').getAttribute('cx'))} ${y-Number(g.querySelector('circle').getAttribute('cy'))})`)}}

async function refresh(){
  try {
    render(await api('/state'));
    const ev=(await api(`/events?after=${eventSeq}`)).events;
    if(ev.length){
      eventSeq=ev.at(-1).seq;
      for(const e of ev){
        const d=document.createElement('div');d.className=`event ${e.level}`;
        d.textContent=`${e.timestamp.slice(11,19)}  ${e.message}`;$('events').prepend(d);
      }
    }
  } catch(e) {
    $('connection').textContent='Disconnected';$('health-dot').className='dot offline';
    $('endpoint').textContent=API;
  }
}
async function refreshLogs(){
  try{const logs=(await api('/logs')).logs, root=$('runs');root.innerHTML='';if(!logs.length){root.innerHTML='<p class="empty">No hardware runs recorded.</p>';return}
    for(const r of logs.slice(0,12)){const d=document.createElement('div');d.className='run';const score=r.manual_score?`${r.manual_score.success?'success':'failure'} · ${r.manual_score.reorientation_deg??'—'}°`:'unscored';d.innerHTML=`<strong>${r.design} · ${r.status}</strong><small>${r.run_id}<br>${score}</small><div class="actions"><button class="secondary download">Data</button><button class="secondary score">Score</button></div>`;d.querySelector('.download').onclick=()=>window.open(`${API}/api/v1/logs/${r.run_id}.jsonl`);d.querySelector('.score').onclick=()=>openScore(r.run_id);root.appendChild(d)}
  }catch(e){toast(e.message,true)}
}
function openScore(id){$('score-run-id').value=id;$('score-dialog').showModal()}

$('plan-select').onchange=()=>showPlan(plans.find(p=>p.file===$('plan-select').value));
$('load-plan').onclick=()=>action(async()=>{await post('/plans/load',{file:$('plan-select').value});toast('Candidate loaded and validated')});
$('home').onclick=()=>{$('home-phrase').value='';$('confirm-home').showModal()};
$('confirm-home-button').onclick=e=>{e.preventDefault();action(async()=>{await post('/home',{confirmation:$('home-phrase').value});$('confirm-home').close();toast('Homing accepted — stay with the hand')})};
$('apply-morph').onclick=()=>action(async()=>{await post('/morphology');toast('Morphology move accepted')});
const motion=()=>({speed_ratio:Number($('speed').value),rate_hz:Number($('rate').value)});
$('pose-open').onclick=()=>action(()=>post('/pose',{name:'open',...motion()}));
$('pose-grip').onclick=()=>action(()=>post('/pose',{name:'grip',...motion()}));
$('run').onclick=()=>action(async()=>{const r=await post('/reorient',motion());latestRun=r.run_id;toast(`Run ${r.run_id} recording`)});
$('stop').onclick=()=>action(()=>post('/stop'));
$('speed').oninput=()=>{$('speed-value').textContent=`${Number($('speed').value).toFixed(2)}×`};
$('refresh-logs').onclick=refreshLogs;
$('save-score').onclick=e=>{e.preventDefault();action(async()=>{const angle=$('score-angle').value;await post(`/logs/${$('score-run-id').value}/score`,{success:$('score-success').value==='true',reorientation_deg:angle===''?null:Number(angle),notes:$('score-notes').value});$('score-dialog').close();await refreshLogs();toast('Manual score saved')})};

$('endpoint').textContent=API;loadPlans();refresh();refreshLogs();setInterval(refresh,750);setInterval(refreshLogs,5000);
