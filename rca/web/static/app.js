"use strict";
/* Indian Road Capacity Analyzer — dashboard */
const $ = (id) => document.getElementById(id);
const S = { options: null, scan: null, accepted: new Set(), cap: null, jobId: null, traffic: null,
            twin: null, twinReq: null, photoFile: null, photoImg: null, trap: [], edgeMode: false,
            videoFile: null, frameImg: null, lineA: [], lineB: [], lineMode: null };
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "—"
  : Number(v).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : msg; } catch (_) {}
    throw new Error(msg);
  }
  return r;
}
const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "twin") prefillTwin();
}));

/* ---------------- options & road form ---------------- */
async function init() {
  try {
    const h = await (await api("/api/health")).json();
    $("health").textContent = h.models["defects.onnx"] && h.models["vehicles.onnx"] ? "● server ready" : "⚠ model file missing";
  } catch (e) { $("health").textContent = "⚠ server not reachable"; }
  S.options = await (await api("/api/options")).json();
  const rt = $("road_type");
  rt.innerHTML = S.options.road_types.map((r) => `<option value="${r.key}">${esc(r.label)} — ${fmt(r.capacity_pcu_h)} PCU/h (${r.basis})</option>`).join("");
  rt.value = "divided_4lane";
  for (const name of ["on_street_parking", "bus_stop", "access_points"]) {
    $(name).innerHTML = S.options.factors[name].map((o) => `<option value="${o.key}">${esc(o.label)} (×${o.factor.toFixed(2)})</option>`).join("");
  }
  $("analysis_date").value = new Date().toISOString().slice(0, 10);
  onRoadType();
  rt.addEventListener("change", onRoadType);
  ["width_m", "on_street_parking", "bus_stop", "access_points", "operating_speed_kmh", "iri", "width_method", "road_name"]
    .forEach((id) => $(id).addEventListener("change", recompute));
  $("volume_pcu_h").addEventListener("change", () => { $("vol_src").textContent = $("volume_pcu_h").value ? "entered by you" : "entered by you only"; S.volumeSource = "entered by user"; recompute(); });
  recompute();
}
function onRoadType() {
  const r = S.options.road_types.find((x) => x.key === $("road_type").value);
  $("width_m").value = r.paved_width_m;
  $("width_hint").textContent = r.divided ? "per direction" : "both directions";
  recompute();
}
function roadDict() {
  const num = (id) => $(id).value === "" ? null : Number($(id).value);
  return { road_name: $("road_name").value, road_type: $("road_type").value, width_m: num("width_m"),
           on_street_parking: $("on_street_parking").value, bus_stop: $("bus_stop").value,
           access_points: $("access_points").value, operating_speed_kmh: num("operating_speed_kmh"),
           iri: num("iri"), width_method: $("width_method").value, volume_pcu_h: num("volume_pcu_h"),
           volume_source: $("volume_pcu_h").value ? (S.volumeSource || "entered by user") : null,
           analysis_date: $("analysis_date").value || null, data_collection_date: $("data_collection_date").value || null };
}
function acceptedObjects() {
  if (!S.scan) return [];
  return S.scan.scan.objects.filter((_, i) => S.accepted.has(i));
}

/* ---------------- capacity ---------------- */
let recomputeTimer = null;
function recompute() { clearTimeout(recomputeTimer); recomputeTimer = setTimeout(doRecompute, 120); }
async function doRecompute() {
  const objs = acceptedObjects();
  const body = { road: roadDict(),
    intervals_m: objs.filter((o) => o.acts_through && o.acts_through.startsWith("F_W")).map((o) => [o.left_m, o.right_m]),
    parking_detected: objs.some((o) => o.acts_through && o.acts_through.startsWith("F_OP")) };
  if (S.scan) refreshActions();
  try { S.cap = await (await postJSON("/api/capacity", body)).json(); renderCap(); }
  catch (e) { $("cap_notes").innerHTML = `<p class="status err">${esc(e.message)}</p>`; }
}
function losBadge(l) { return l ? `<span class="los los-${l}">${l}</span>` : "—"; }
function renderCap() {
  const c = S.cap;
  const k = [
    ["Capacity — clear road", fmt(c.capacity_hcm_pcu_h), `PCU/h, ${c.basis} · Indo-HCM`],
    ["Capacity — with defects", fmt(c.capacity_defect_pcu_h), `PCU/h · F_W ${fmt(c.factors.F_W, 3)} × F_R ${fmt(c.factors.F_R, 3)}`, true],
    ["Capacity loss", fmt(c.capacity_loss_pct, 1) + " %", "due to accepted obstructions"],
    ["Design service volume", fmt(c.dsv_defect_pcu_h), `PCU/h (${fmt(c.dsv_ratio * 100)} % of C)`],
    ["V/C", c.vc_defect == null ? "—" : fmt(c.vc_defect, 2), c.volume_pcu_h == null ? "enter a volume" : `${fmt(c.volume_pcu_h)} PCU/h · ${esc(c.volume_source)}`],
    ["LOS (clear → defects)", c.los_defect ? `${losBadge(c.los_hcm)} → ${losBadge(c.los_defect)}` : "—", `Indo-HCM Table ${c.los_table}`],
  ];
  $("kpis").innerHTML = k.map(([kk, v, s, hl]) => `<div class="kpi${hl ? " hl" : ""}"><div class="k">${kk}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  $("steps").querySelector("tbody").innerHTML = c.steps.map((s) =>
    `<tr><td>${esc(s.step)}</td><td><b>${s.unit === "-" ? fmt(s.value, 2) : fmt(s.value) + " " + s.unit}</b></td><td>${esc(s.source)}</td><td>${esc(s.detail)}</td></tr>`).join("");
  $("cap_notes").innerHTML = c.notes.length ? "<ul>" + c.notes.map((n) => `<li>${esc(n)}</li>`).join("") + "</ul>" : "<p>None.</p>";
  renderLosScale();
}
function renderLosScale() {
  const c = S.cap, key = { "5.6": "undivided", "5.7": "divided", "2.9": "intermediate", "2.10": "single" }[c.los_table];
  const bands = S.options.los_tables[key];
  const cols = { A: "var(--los-a)", B: "var(--los-b)", C: "var(--los-c)", D: "var(--los-d)", E: "var(--los-e)", F: "var(--los-f)" };
  let html = `<div class="losbar">` + bands.map(([l, u]) => `<div style="background:${cols[l]}">${l} ≤${u}</div>`).join("") + `<div style="background:${cols.F}">F</div>`;
  const pos = (vc) => { // piecewise position across equal-width bands
    let lo = 0; for (let i = 0; i < bands.length; i++) { const up = bands[i][1]; if (vc <= up) return (i + (vc - lo) / (up - lo)) / (bands.length + 1); lo = up; }
    return Math.min((bands.length + Math.min((vc - 1) / 0.5, 0.95)) / (bands.length + 1), 0.99); };
  if (c.vc_hcm != null) html += `<div class="marker" style="left:${pos(c.vc_hcm) * 100}%;background:#2563eb"><span>clear ${fmt(c.vc_hcm, 2)}</span></div>`;
  if (c.vc_defect != null) html += `<div class="marker" style="left:${pos(c.vc_defect) * 100}%"><span>defects ${fmt(c.vc_defect, 2)}</span></div>`;
  html += "</div>";
  html += c.vc_defect == null ? `<p class="hint">Enter a traffic volume (tab 1) or use the video count (tab 2) to see V/C and LOS.</p>`
    : `<p class="hint">V/C bands of Indo-HCM Table ${c.los_table}. Volume source: ${esc(c.volume_source)}.</p>`;
  $("los_scale").innerHTML = html;
}

/* ---------------- photo & road edges ---------------- */
const pc = $("photo_canvas");
$("photo").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) loadPhoto(f, []); });
document.querySelectorAll("button.sample").forEach((b) => b.addEventListener("click", async () => {
  const blob = await (await api(b.dataset.src)).blob();
  loadPhoto(new File([blob], b.dataset.src.split("/").pop(), { type: "image/jpeg" }), b.dataset.trap ? JSON.parse(b.dataset.trap) : []);
}));
function loadPhoto(f, trap) {
  S.photoFile = f; S.trap = trap;
  const img = new Image();
  img.onload = () => { S.photoImg = img; pc.width = img.naturalWidth; pc.height = img.naturalHeight; drawPhoto();
    $("btn_scan").disabled = false; $("btn_edges").disabled = false; $("btn_clear_edges").disabled = false; };
  img.src = URL.createObjectURL(f);
}
$("btn_edges").addEventListener("click", () => { S.edgeMode = !S.edgeMode; if (S.edgeMode) S.trap = []; $("btn_edges").classList.toggle("on", S.edgeMode); drawPhoto(); });
$("btn_clear_edges").addEventListener("click", () => { S.trap = []; S.edgeMode = false; $("btn_edges").classList.remove("on"); drawPhoto(); });
function canvasPoint(cv, ev) {
  const r = cv.getBoundingClientRect();
  return [(ev.clientX - r.left) * cv.width / r.width, (ev.clientY - r.top) * cv.height / r.height];
}
pc.addEventListener("click", (ev) => {
  if (!S.edgeMode || S.trap.length >= 4) return;
  S.trap.push(canvasPoint(pc, ev));
  if (S.trap.length === 4) { S.edgeMode = false; $("btn_edges").classList.remove("on"); }
  drawPhoto();
});
function drawPhoto() {
  if (!S.photoImg) return;
  const ctx = pc.getContext("2d"); ctx.drawImage(S.photoImg, 0, 0);
  const lw = Math.max(2, pc.width / 400), names = ["near-left", "near-right", "far-right", "far-left"];
  ctx.strokeStyle = "#22c55e"; ctx.fillStyle = "#22c55e"; ctx.lineWidth = lw; ctx.font = `${Math.round(lw * 7)}px sans-serif`;
  ctx.beginPath(); S.trap.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (S.trap.length === 4) ctx.closePath(); ctx.stroke();
  S.trap.forEach(([x, y], i) => { ctx.beginPath(); ctx.arc(x, y, lw * 3, 0, 7); ctx.fill(); ctx.fillText(names[i], x + lw * 4, y - lw * 3); });
  $("edge_help").textContent = S.edgeMode ? `Click the ${names[S.trap.length]} road edge.` :
    S.trap.length === 4 ? "Road edges marked: widths are perspective-corrected." :
    "Optional: click 4 points — near-left, near-right, far-right, far-left road edge — for perspective-corrected widths.";
}
$("btn_scan").addEventListener("click", async () => {
  if (S.trap.length && S.trap.length !== 4) { $("scan_status").textContent = "Mark all 4 road-edge points or clear them."; return; }
  const fd = new FormData();
  fd.append("image", S.photoFile); fd.append("road", JSON.stringify(roadDict()));
  if (S.trap.length === 4) fd.append("trapezoid", JSON.stringify(S.trap));
  fd.append("depth", $("use_depth").checked ? "1" : "0");
  $("btn_scan").disabled = true; $("scan_status").className = "status"; $("scan_status").textContent = "Detecting…";
  try {
    S.scan = await (await api("/api/scan", { method: "POST", body: fd })).json();
    S.accepted = new Set(S.scan.scan.objects.map((_, i) => i).filter((i) => S.scan.scan.objects[i].class !== "vehicle"));
    renderScan(); recompute(); renderActions(S.scan.legacy); renderScanExports();
    $("scan_status").textContent = `Done: ${S.scan.scan.objects.filter((o) => o.class !== "vehicle").length} obstruction(s). Check them below.`;
  } catch (e) { $("scan_status").className = "status err"; $("scan_status").textContent = e.message; }
  $("btn_scan").disabled = false;
});
function renderScan() {
  const sc = S.scan.scan; $("scan_results").hidden = false;
  $("annotated").src = S.scan.annotated_url + "?t=" + Date.now();
  const rows = sc.objects.map((o, i) => o.class === "vehicle" ? "" :
    `<tr class="${S.accepted.has(i) ? "" : "off"}"><td><input type="checkbox" data-i="${i}" ${S.accepted.has(i) ? "checked" : ""}></td>
     <td>${esc(o.class.replace("_", " "))}${o.irc82_size ? ` <span class="hint">(${o.irc82_size}, IRC:82)</span>` : ""}</td>
     <td>${o.conf.toFixed(2)}</td><td>${o.left_m.toFixed(2)} – ${o.right_m.toFixed(2)}</td><td>${o.width_m.toFixed(2)} m</td>
     <td>${esc(o.acts_through || "off carriageway")}</td>
     <td>${esc(o.found_by || "")}${o.depth_indicator ? `<br><span class="hint">depth: ${esc(o.depth_indicator.depth_class)} (${o.depth_indicator.depth_score})</span>` : ""}</td></tr>`).join("");
  $("obj_table").querySelector("tbody").innerHTML = rows || `<tr><td colspan="7">No obstructions detected.</td></tr>`;
  $("obj_table").querySelectorAll("input[type=checkbox]").forEach((cb) => cb.addEventListener("change", () => {
    const i = Number(cb.dataset.i); cb.checked ? S.accepted.add(i) : S.accepted.delete(i);
    cb.closest("tr").classList.toggle("off", !cb.checked); recompute(); refreshActions(); }));
  const lm = sc.lane_markings;
  if (lm && lm.available) {
    const rows = lm.lines.map((l) => `<tr><td>L${l.id}</td><td>${l.type}</td><td>${l.colour}</td><td>${Math.round(l.painted_share * 100)} %</td><td>${fmt(l.contrast_ratio, 2)}</td><td><b>${fmt(l.score)}</b> (${esc(l.condition)})</td></tr>`).join("");
    $("lane_card").innerHTML = `<b>Lane markings:</b> ${lm.n_lines} line(s) · road score <b>${fmt(lm.visibility_index)}/100</b> (${esc(lm.condition)})`
      + (lm.n_lines ? `<table class="tbl" style="margin-top:6px"><thead><tr><th>Line</th><th>Type</th><th>Colour</th><th>Painted</th><th>Contrast</th><th>Score</th></tr></thead><tbody>${rows}</tbody></table>` : "")
      + `<br><span class="hint">${esc(lm.status)}</span>`;
  } else {
    $("lane_card").innerHTML = `<b>Lane markings:</b> ${esc((lm && lm.reason) || "not assessed")}`;
  }
  $("scan_notes").innerHTML = sc.notes.length ? "<ul>" + sc.notes.map((n) => `<li>${esc(n)}</li>`).join("") + "</ul>" : "";
}

/* ---------------- video ---------------- */
const vc = $("video_canvas");
$("video").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  S.videoFile = f; S.lineA = []; S.lineB = [];
  $("video_status").className = "status"; $("video_status").textContent = "Reading first frame…";
  const fd = new FormData(); fd.append("video", f);
  try {
    const info = await (await api("/api/video/frame", { method: "POST", body: fd })).json();
    const img = new Image();
    img.onload = () => { S.frameImg = img; vc.width = img.naturalWidth; vc.height = img.naturalHeight; drawFrame(); };
    img.src = info.image;
    $("video_status").textContent = `${info.width}×${info.height}, ${fmt(info.fps, 1)} fps, ${fmt(info.frames / info.fps)} s. Draw line A.`;
    $("btn_line_a").disabled = $("btn_line_b").disabled = false;
  } catch (err) { $("video_status").className = "status err"; $("video_status").textContent = err.message; }
});
$("btn_line_a").addEventListener("click", () => { S.lineMode = "A"; S.lineA = []; drawFrame(); });
$("btn_line_b").addEventListener("click", () => { S.lineMode = "B"; S.lineB = []; drawFrame(); });
vc.addEventListener("click", (ev) => {
  if (!S.lineMode) return;
  const L = S.lineMode === "A" ? S.lineA : S.lineB;
  L.push(canvasPoint(vc, ev));
  if (L.length === 2) S.lineMode = null;
  drawFrame();
});
function drawFrame() {
  if (!S.frameImg) return;
  const ctx = vc.getContext("2d"); ctx.drawImage(S.frameImg, 0, 0);
  const lw = Math.max(2, vc.width / 300);
  [[S.lineA, "#22c55e", "A"], [S.lineB, "#f59e0b", "B"]].forEach(([L, col, n]) => {
    ctx.strokeStyle = ctx.fillStyle = col; ctx.lineWidth = lw; ctx.font = `bold ${Math.round(lw * 8)}px sans-serif`;
    L.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, lw * 2.5, 0, 7); ctx.fill(); });
    if (L.length === 2) { ctx.beginPath(); ctx.moveTo(...L[0]); ctx.lineTo(...L[1]); ctx.stroke(); ctx.fillText(n, L[0][0] + 6, L[0][1] - 8); }
  });
  $("btn_line_a").classList.toggle("on", S.lineMode === "A"); $("btn_line_b").classList.toggle("on", S.lineMode === "B");
  $("btn_count").disabled = S.lineA.length !== 2;
}
$("btn_count").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("video", S.videoFile); fd.append("line_a", JSON.stringify(S.lineA)); fd.append("road_type", $("road_type").value);
  if (S.lineB.length === 2) {
    if (!$("trap_m").value) { $("video_status").className = "status err"; $("video_status").textContent = "Enter the distance between line A and line B."; return; }
    fd.append("line_b", JSON.stringify(S.lineB)); fd.append("trap_length_m", $("trap_m").value);
    if ($("speed_limit").value) fd.append("speed_limit_kmh", $("speed_limit").value);
  } else if ($("speed_limit").value) { $("video_status").className = "status err"; $("video_status").textContent = "The speed check needs line B and the trap length."; return; }
  $("btn_count").disabled = true; $("video_status").className = "status"; $("video_status").textContent = "Uploading…";
  document.querySelector(".progress").hidden = false; $("prog_bar").style.width = "0";
  try {
    const { job_id } = await (await api("/api/video", { method: "POST", body: fd })).json();
    S.jobId = job_id; pollJob();
  } catch (e) { $("video_status").className = "status err"; $("video_status").textContent = e.message; $("btn_count").disabled = false; }
});
async function pollJob() {
  try {
    const j = await (await api(`/api/jobs/${S.jobId}`)).json();
    $("prog_bar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
    if (j.status === "done") { S.traffic = j.result; renderTraffic(); $("video_status").textContent = "Counting finished."; $("btn_count").disabled = false; return; }
    if (j.status === "error") { throw new Error(j.error); }
    $("video_status").textContent = j.status === "queued" ? "Waiting for the previous video to finish…" : `Counting… ${Math.round((j.progress || 0) * 100)} %`;
    setTimeout(pollJob, 1500);
  } catch (e) { $("video_status").className = "status err"; $("video_status").textContent = e.message; $("btn_count").disabled = false; }
}
function renderTraffic() {
  const t = S.traffic; $("video_results").hidden = false;
  const divided = S.options.road_types.find((r) => r.key === $("road_type").value).divided;
  let html = `<p>${t.total_vehicles} vehicles crossed line A in ${fmt(t.duration_s)} s.</p><table class="tbl"><thead><tr><th>Direction</th><th>Vehicles</th><th>veh/h</th><th>PCU/h</th><th>Speed (SMS)</th><th>Density</th><th></th></tr></thead><tbody>`;
  let total = 0;
  for (const [d, v] of Object.entries(t.directions)) {
    total += v.flow_pcu_h;
    html += `<tr><td>${d === "down" ? "towards camera" : "away from camera"}</td><td>${Object.entries(v.counts).map(([k, n]) => `${k.replace("_", " ")} ${n}`).join(", ")}</td>
      <td>${fmt(v.flow_veh_h)}</td><td><b>${fmt(v.flow_pcu_h)}</b></td><td>${v.space_mean_speed_kmh ? fmt(v.space_mean_speed_kmh, 1) + " km/h" : "—"}</td>
      <td>${v.density_pcu_km ? fmt(v.density_pcu_km, 1) + " PCU/km" : "—"}</td>
      <td>${divided ? `<button class="ghost use" data-v="${v.flow_pcu_h}" data-d="${d}">Use as volume</button>` : ""}</td></tr>`;
  }
  html += "</tbody></table>";
  if (!divided) html += `<p><button class="ghost use" data-v="${total.toFixed(1)}" data-d="both">Use both directions (${fmt(total)} PCU/h) as volume</button> <span class="hint">undivided road: capacity is for both directions</span></p>`;
  const op = Object.values(t.directions).map((v) => v.operating_speed).find((o) => o && o.operating_speed_kmh);
  if (op) html += `<p>Operating speed of free cars: <b>${fmt(op.operating_speed_kmh, 1)} km/h</b> (${op.n_free_cars} cars)${op.warning ? ` <span class="hint">${esc(op.warning)}</span>` : ""}</p>`;
  const sc = t.speed_check;
  if (sc) {
    html += `<h2 style="margin-top:12px">Speed check (limit ${sc.speed_limit_kmh} km/h)</h2><p>${sc.over_limit.length} of ${sc.vehicles_with_speed} timed vehicles over the limit.</p>`;
    if (sc.over_limit.length) html += `<table class="tbl"><thead><tr><th>Vehicle</th><th>Speed</th><th>Over by</th><th>Time in video</th><th>Evidence</th><th>Status</th></tr></thead><tbody>` +
      sc.over_limit.map((e) => `<tr><td>#${e.track_id} ${esc(e.class)}</td><td>${fmt(e.speed_kmh, 1)} km/h</td><td>${fmt(e.over_limit_by_kmh, 1)}</td><td>${fmt(e.time_in_video_s, 1)} s</td>
        <td>${e.evidence_image ? `<a href="${e.evidence_image}" target="_blank"><img src="${e.evidence_image}" style="height:48px;border-radius:4px"></a>` : "—"}</td><td>${esc(e.verification_status)}</td></tr>`).join("") + "</tbody></table>";
    html += `<p class="hint">${esc(sc.note)} ${esc(sc.over_limit[0] ? sc.over_limit[0].applicable_section : "")}</p>`;
  }
  html += "<ul class='notes'>" + t.notes.map((n) => `<li>${esc(n)}</li>`).join("") + "</ul>";
  $("video_summary").innerHTML = html;
  $("video_summary").querySelectorAll("button.use").forEach((b) => b.addEventListener("click", () => {
    $("volume_pcu_h").value = Math.round(Number(b.dataset.v));
    S.volumeSource = `video count (${b.dataset.d === "both" ? "both directions" : b.dataset.d === "down" ? "towards camera" : "away from camera"}), confirmed by user`;
    $("vol_src").textContent = S.volumeSource; recompute();
  }));
  if (t.annotated_url) { const v = $("ann_video"); v.src = t.annotated_url; v.hidden = false; }
}

/* ---------------- digital twin ---------------- */
function prefillTwin() {
  if (!$("tw_demand").value && $("volume_pcu_h").value) $("tw_demand").value = $("volume_pcu_h").value;
  if (!$("tw_vf").value) {
    const v = $("operating_speed_kmh").value || (S.traffic && Object.values(S.traffic.directions).map((d) => d.operating_speed && d.operating_speed.operating_speed_kmh).find(Boolean));
    if (v) $("tw_vf").value = Math.round(v);
  }
}
$("btn_twin").addEventListener("click", async () => {
  if (!S.cap) return;
  const demand = Number($("tw_demand").value), vf = Number($("tw_vf").value);
  if (!$("tw_demand").value || !vf) { $("twin_help").textContent = "Enter the demand (PCU/h) and free-flow speed (km/h) from your survey."; $("twin_help").className = "status err"; return; }
  $("twin_help").className = "hint"; $("twin_help").textContent = "Running…";
  S.twinReq = { capacity_clear_pcu_h: S.cap.capacity_hcm_pcu_h, capacity_defect_pcu_h: Math.max(S.cap.capacity_defect_pcu_h, 1),
    vf_kmh: vf, demand_pcu_h: demand, length_m: Number($("tw_len").value), obstruction_at: Number($("tw_at").value),
    obstruction_length_m: Number($("tw_olen").value), duration_s: Number($("tw_dur").value), demand_profile: $("tw_prof").value };
  try {
    S.twin = await (await postJSON("/api/twin", S.twinReq)).json();
    $("twin_help").textContent = S.twin.summary.note + (S.twin.summary.breakdown ? " " + S.twin.summary.queue_note : "");
    renderTwin();
  } catch (e) { $("twin_help").className = "status err"; $("twin_help").textContent = e.message; }
});
function ramp(u) { // green → yellow → red → dark red
  const stops = [[26, 152, 80], [254, 224, 139], [215, 48, 39], [74, 0, 0]];
  u = Math.max(0, Math.min(1, u)) * (stops.length - 1); const i = Math.min(Math.floor(u), stops.length - 2), f = u - i;
  return stops[i].map((c, j) => Math.round(c + f * (stops[i + 1][j] - c)));
}
function heatmap(cv, sim, kjam) {
  cv.width = sim.x_m.length; cv.height = sim.times_s.length;
  const ctx = cv.getContext("2d"), im = ctx.createImageData(cv.width, cv.height);
  sim.density_pcu_km.forEach((row, t) => row.forEach((k, x) => {
    const [r, g, b] = ramp(k / kjam), p = ((cv.height - 1 - t) * cv.width + x) * 4;
    im.data[p] = r; im.data[p + 1] = g; im.data[p + 2] = b; im.data[p + 3] = 255; }));
  ctx.putImageData(im, 0, 0); cv.style.imageRendering = "pixelated";
}
let anim = null;
function renderTwin() {
  const T = S.twin; $("twin_out").hidden = false;
  const c = T.clear, o = T.obstructed, sm = T.summary;
  $("twin_kpis").innerHTML = [
    ["Mean delay, clear", fmt(c.mean_delay_s_per_pcu, 1) + " s", "per PCU vs free flow"],
    ["Mean delay, obstructed", fmt(o.mean_delay_s_per_pcu, 1) + " s", `+${fmt(sm.extra_delay_s_per_pcu, 1)} s per PCU`, true],
    ["Max queue", fmt(o.max_queue_m) + " m", sm.breakdown ? "queue keeps growing" : "no standing queue"],
    ["Throughput (last 5 min)", fmt(o.throughput_pcu_h_last5min) + " PCU/h", `clear road ${fmt(c.throughput_pcu_h_last5min)}`],
  ].map(([k, v, s, hl]) => `<div class="kpi${hl ? " hl" : ""}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  heatmap($("hm_clear"), c, c.jam_density_pcu_km); heatmap($("hm_obs"), o, c.jam_density_pcu_km);
  $("hm_legend").innerHTML = `0 <div class="ramp"></div> ${fmt(c.jam_density_pcu_km)} PCU/km (jam; yellow ≈ critical density, i.e. at capacity) · x: distance along road → · y: time ↑`;
  $("tw_time").max = c.times_s.length - 1; $("tw_time").value = 0;
  startAnimation();
}
function startAnimation() {
  if (anim) cancelAnimationFrame(anim.raf);
  const T = S.twin, L = T.clear.inputs.length_m, scale = 1 / 3; // one dot = 3 PCU
  const mk = (sim) => ({ sim, dots: [], acc: 0, obs: sim.inputs.obstruction_m });
  anim = { roads: [mk(T.clear), mk(T.obstructed)], idx: 0, sub: 0, playing: true, raf: 0 };
  $("btn_play").textContent = "❚❚ Pause";
  const cvs = [$("road_clear"), $("road_obs")];
  const step = () => {
    const n = T.clear.times_s.length;
    if (anim.playing) {
      anim.sub += 1;
      if (anim.sub >= 6) { anim.sub = 0; anim.idx = Math.min(anim.idx + 1, n - 1); $("tw_time").value = anim.idx; if (anim.idx === n - 1) anim.playing = false; }
    }
    const i = anim.idx, dtSim = (T.clear.times_s[1] - T.clear.times_s[0]) / 6;
    anim.roads.forEach((rd, r) => {
      const sp = rd.sim.speed_kmh[i], dx = rd.sim.dx_m;
      if (anim.playing) {
        const v0 = sp[0] / 3.6;
        rd.acc += rd.sim.inputs.demand_pcu_h / 3600 * dtSim * scale;
        const gap = 8 / scale / Math.max(rd.sim.density_pcu_km[i][0], 1) * 1000 / 8; // spacing ≈ 1/k
        while (rd.acc >= 1 && (!rd.dots.length || rd.dots[rd.dots.length - 1] > Math.min(gap, 60)) && v0 > 1) { rd.dots.push(0); rd.acc -= 1; }
        rd.dots = rd.dots.map((x) => x + (sp[Math.min(Math.floor(x / dx), sp.length - 1)] / 3.6) * dtSim).filter((x) => x < L);
      }
      drawRoad(cvs[r], rd, sp, L);
    });
    const t = T.clear.times_s[i]; $("tw_clock").textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
    anim.raf = requestAnimationFrame(step);
  };
  anim.raf = requestAnimationFrame(step);
}
function drawRoad(cv, rd, sp, L) {
  const w = cv.clientWidth || 800; cv.width = w; const h = cv.height, ctx = cv.getContext("2d");
  ctx.fillStyle = "#3a3f46"; ctx.fillRect(0, 10, w, h - 20);
  const n = sp.length, vf = rd.sim.inputs.vf_kmh;
  for (let j = 0; j < n; j++) { const [r, g, b] = ramp(1 - sp[j] / vf); ctx.fillStyle = `rgba(${r},${g},${b},.35)`; ctx.fillRect(j * w / n, 10, w / n + 1, h - 20); }
  ctx.setLineDash([14, 12]); ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke(); ctx.setLineDash([]);
  if (rd.obs) { const x0 = rd.obs[0] / L * w, x1 = Math.max(rd.obs[1] / L * w, x0 + 6);
    ctx.fillStyle = "#f59e0b"; ctx.fillRect(x0, 10, x1 - x0, (h - 20) * 0.45); ctx.fillStyle = "#111"; ctx.font = "11px sans-serif"; ctx.fillText("obstruction", x0, 9 + 0); }
  ctx.fillStyle = "#f8fafc";
  rd.dots.forEach((x, k) => { const y = k % 2 ? h * 0.66 : h * 0.34; ctx.fillRect(x / L * w - 3, y - 3, 7, 6); });
}
$("btn_play").addEventListener("click", () => { if (!anim) return; anim.playing = !anim.playing; if (anim.idx >= S.twin.clear.times_s.length - 1) { startAnimation(); return; } $("btn_play").textContent = anim.playing ? "❚❚ Pause" : "▶ Play"; });
$("tw_time").addEventListener("input", (e) => { if (anim) { anim.idx = Number(e.target.value); anim.playing = false; $("btn_play").textContent = "▶ Play"; } });

/* ---------------- report ---------------- */
$("btn_report").addEventListener("click", async () => {
  $("report_status").className = "status"; $("report_status").textContent = "Building PDF…";
  const body = { road: roadDict(), scan_id: S.scan ? S.scan.id : null, accepted: S.scan ? [...S.accepted] : null,
                 job_id: $("rep_video").checked && S.traffic ? S.jobId : null, twin: $("rep_twin").checked && S.twin ? S.twinReq : null };
  try {
    const blob = await (await postJSON("/api/report", body)).blob();
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `road_capacity_${($("road_name").value || "report").replace(/[^a-z0-9]+/gi, "_")}.pdf`; a.click();
    $("report_status").textContent = "Report downloaded.";
  } catch (e) { $("report_status").className = "status err"; $("report_status").textContent = e.message; }
});

init();


/* ---------------- recommended actions (earlier version's rules) ---------------- */
const SEV = { URGENT: "#b91c1c", ROUTINE: "#b45309", MONITOR: "#15803d", INVESTIGATE: "#6b7280" };
function renderActions(leg) {
  if (!leg) return;
  $("actions_card").hidden = false;
  const g = leg.overall_guidance || {};
  $("guidance").innerHTML = `<b>Overall: ${esc(g.band || "-")}</b> — ${esc(g.action || "")} <span class="hint">(capacity loss ${fmt(leg.capacity_loss_pct, 1)} % of base capacity; ${esc(g.basis || "")})</span>`;
  const pd = leg.per_defect || {};
  $("actions_table").querySelector("tbody").innerHTML = Object.keys(pd).length ? Object.entries(pd).map(([k, d]) =>
    `<tr><td>${esc(k.replace("_", " "))}</td><td>${d.count}</td><td>${fmt(d.factor, 3)}</td><td>${fmt(d.capacity_loss_pct, 1)} %</td>
     <td><b style="color:${SEV[d.severity] || "#333"}">${esc(d.severity)}</b><br><span class="hint">${esc(d.severity_basis)}</span></td>
     <td>${esc(d.action)}</td><td class="hint">${esc(d.code_ref)}</td></tr>`).join("") : `<tr><td colspan="7">No obstructions accepted — no action required.</td></tr>`;
  const r = (pd.pothole || {}).rectification, ds = (pd.pothole || {}).depth_summary;
  $("rectification").innerHTML = (r ? `<p><b>Pothole repair — ${esc(r.category)}:</b> ${esc(r.method)}<br>Materials: ${esc(r.materials)}<br>Priority: ${esc(r.priority)} · ${esc(r.irc_reference)}<br>${esc(r.work_zone_note)} ${esc(r.escalation_note || "")}</p>` : "")
    + (ds ? `<p><b>Depth indicator:</b> ${Object.entries(ds.counts).map(([k, v]) => `${k} ${v}`).join(", ")}. ${esc(ds.note)}</p>` : "");
}
let actTimer = null;
function refreshActions() {
  clearTimeout(actTimer);
  actTimer = setTimeout(async () => {
    try {
      const leg = await (await postJSON("/api/legacy", { scan_id: S.scan.id, road: roadDict(), accepted: [...S.accepted] })).json();
      renderActions(leg);
    } catch (_) {}
  }, 250);
}

/* ---------------- exports ---------------- */
const EXPORTS = [["department-report.pdf", "Department report (PDF)"], ["defects.xlsx", "Defect log (Excel)"],
  ["roadrunner.xodr", "RoadRunner .xodr"], ["roadrunner-ideal.xodr", "RoadRunner ideal .xodr"],
  ["roadrunner-capacity.json", "Capacity JSON"], ["bundle.zip", "Digital-twin bundle (MATLAB)"]];
async function download(kind, body, fname) {
  const blob = await (await postJSON(`/api/export/${kind}`, body)).blob();
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = fname; a.click();
}
function exportButtons(el, bodyFn, stem) {
  el.innerHTML = EXPORTS.map(([k, label]) => `<button class="ghost" data-k="${k}">${label}</button>`).join("");
  el.querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true; const t = b.textContent; b.textContent = "Preparing…";
    try { await download(b.dataset.k, bodyFn(), `${stem()}_${b.dataset.k}`); } catch (e) { alert(e.message); }
    b.disabled = false; b.textContent = t;
  }));
}
const stemName = () => ($("road_name").value || "road").replace(/[^a-z0-9]+/gi, "_");
function renderScanExports() {
  exportButtons($("scan_exports"), () => ({ scan_id: S.scan.id, road: roadDict(), accepted: [...S.accepted], site_label: $("road_name").value || null }), stemName);
}

/* ---------------- corridor ---------------- */
$("btn_corridor").addEventListener("click", async () => {
  const files = $("corr_files").files;
  if (!files.length) { $("corr_status").textContent = "Choose the photos first (in order along the road)."; return; }
  const fd = new FormData();
  [...files].forEach((f) => fd.append("images", f));
  fd.append("road", JSON.stringify(roadDict())); fd.append("spacing_m", $("corr_spacing").value || "100");
  fd.append("start_chainage_m", $("corr_start").value || "0"); fd.append("depth", $("use_depth").checked ? "1" : "0");
  $("btn_corridor").disabled = true; $("corr_status").className = "status"; $("corr_status").textContent = `Analysing ${files.length} photo(s)…`;
  try {
    const c = await (await api("/api/corridor", { method: "POST", body: fd })).json();
    S.corridor = c; $("corr_out").hidden = false;
    const sm = c.summary;
    $("corr_kpis").innerHTML = [["Bottleneck capacity", fmt(sm.bottleneck_capacity_pcu_h) + " PCU/h", `at ${esc(sm.worst_image_or_frame || "-")}`, true],
      ["Worst loss", fmt(sm.worst_capacity_loss_pct, 1) + " %", "of base capacity"], ["Median loss", fmt(sm.median_capacity_loss_pct, 1) + " %", "how widespread"],
      ["Sections", c.sections.length, c.errors.length ? `${c.errors.length} unreadable` : "all analysed"]]
      .map(([k, v, s2, hl]) => `<div class="kpi${hl ? " hl" : ""}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s2}</div></div>`).join("");
    $("corr_table").querySelector("tbody").innerHTML = c.sections.map((x) => `<tr><td>${fmt(x.chainage_m)} m</td><td><a href="${x.annotated_url}" target="_blank">${esc(x.image)}</a></td>
      <td>${fmt(x.capacity_hcm_pcu_h)}</td><td><b>${fmt(x.reduced_capacity_pcu_h)}</b></td><td>${fmt(x.capacity_loss_pct, 1)} %</td><td>${esc(x.defects_found.join(", ") || "none")}</td><td>${x.lane_score == null ? "—" : fmt(x.lane_score)}</td></tr>`).join("");
    exportButtons($("corr_exports"), () => ({ corridor_id: S.corridor.id, road: roadDict(), site_label: $("road_name").value || null }), () => stemName() + "_corridor");
    $("corr_status").textContent = "Done. Road edges are not marked in corridor mode: widths assume the photo spans the carriageway.";
  } catch (e) { $("corr_status").className = "status err"; $("corr_status").textContent = e.message; }
  $("btn_corridor").disabled = false;
});

/* ---------------- field validation ---------------- */
function renderField(r) {
  const A = r.method_A || {}, B = r.method_B || {};
  let h = `<p>${r.vehicles} vehicles${r.source ? ` · ${esc(r.source)}` : ""}.</p><table class="tbl"><thead><tr><th>Method</th><th>Capacity</th><th>Details</th></tr></thead><tbody>`;
  h += `<tr><td>${esc(A.method || "A")}</td><td><b>${A.capacity_pcu_h == null ? "—" : fmt(A.capacity_pcu_h) + " PCU/h"}</b></td><td>operating speed ${A.operating_speed_kmh == null ? "—" : fmt(A.operating_speed_kmh, 1) + " km/h"} (${A.n_free_cars || 0} free cars) ${esc(A.warning || "")}</td></tr>`;
  h += `<tr><td>${esc(B.method || "B")}</td><td><b>${B.capacity_pcu_h == null ? "—" : fmt(B.capacity_pcu_h) + " PCU/h"}</b></td><td>${B.free_speed_kmh ? `v_f ${fmt(B.free_speed_kmh, 1)} km/h, k_j ${fmt(B.jam_density_pcu_km, 0)} PCU/km, R² ${fmt(B.r2, 2)}. ` : ""}${esc(B.warning || "")}</td></tr>`;
  h += `<tr><td>Peak 5-min flow</td><td>${r.peak_5min_flow_pcu_h == null ? "—" : fmt(r.peak_5min_flow_pcu_h) + " PCU/h"}</td><td>demand, not capacity, unless a queue stood upstream (method C)</td></tr></tbody></table>`;
  if (r.periods && r.periods.length) h += `<details><summary>${r.periods.length} five-minute periods</summary><table class="tbl"><thead><tr><th>Start (s)</th><th>Vehicles</th><th>PCU/h</th><th>SMS km/h</th><th>PCU/km</th></tr></thead><tbody>` +
    r.periods.map((p) => `<tr><td>${fmt(p.start_s)}</td><td>${p.vehicles}</td><td>${fmt(p.flow_pcu_h)}</td><td>${fmt(p.sms_kmh, 1)}</td><td>${fmt(p.density_pcu_km, 1)}</td></tr>`).join("") + "</tbody></table></details>";
  h += (r.warnings || []).map((w) => `<p class="hint">${esc(w)}</p>`).join("");
  $("f_out").innerHTML = h;
}
$("btn_f_csv").addEventListener("click", async () => {
  const f = $("f_csv").files[0]; if (!f) { $("f_status").textContent = "Choose a vehicles.csv first."; return; }
  const fd = new FormData(); fd.append("file", f); fd.append("trap_m", $("f_trap").value); fd.append("road_type", $("road_type").value); fd.append("pcu_method", $("f_pcu").value);
  $("f_status").className = "status"; $("f_status").textContent = "Analysing…";
  try { renderField(await (await api("/api/field/analyse", { method: "POST", body: fd })).json()); $("f_status").textContent = ""; }
  catch (e) { $("f_status").className = "status err"; $("f_status").textContent = e.message; }
});
$("btn_f_video").addEventListener("click", async () => {
  if (!S.jobId || !S.traffic) { $("f_status").textContent = "Count a video in tab 2 first (with line B and trap length)."; return; }
  $("f_status").className = "status"; $("f_status").textContent = "Analysing…";
  try { renderField(await (await postJSON("/api/field/from-video", { job_id: S.jobId, road_type: $("road_type").value, pcu_method: $("f_pcu").value })).json()); $("f_status").textContent = ""; }
  catch (e) { $("f_status").className = "status err"; $("f_status").textContent = e.message; }
});
$("btn_f_compare").addEventListener("click", async () => {
  const pairs = $("f_pairs").value.split("\n").map((l) => l.split(/[,\t ]+/).filter(Boolean).map(Number)).filter((p) => p.length >= 2 && p.every((v) => !Number.isNaN(v)));
  if (!pairs.length) { $("f_cmp").textContent = "Enter at least one line: predicted, observed."; return; }
  try {
    const m = await (await postJSON("/api/field/compare", { predicted: pairs.map((p) => p[0]), observed: pairs.map((p) => p[1]) })).json();
    $("f_cmp").className = "status"; $("f_cmp").innerHTML = `<b>n = ${m.n}</b> · MAPE <b>${fmt(m.mape_pct, 1)} %</b> · RMSE ${fmt(m.rmse, 1)} · bias ${fmt(m.bias, 1)} (predicted − observed)${m.note ? ` <br><span class="hint">${esc(m.note)}</span>` : ""}`;
  } catch (e) { $("f_cmp").className = "status err"; $("f_cmp").textContent = e.message; }
});
