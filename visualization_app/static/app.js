"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  files: [],
  file: null,       // current file path
  columns: [],
  kind: null,
  rows: [],         // all parsed rows
  view: [],         // indices into rows after filtering
  pos: 0,           // position within view
  filters: {},      // columnName -> selected value ("" = all)
  search: "",
  mismatchOnly: false,
  failOnly: false,
};

// The three taxonomy axes (+ image-level nature for COCO/BIG-5). Rendered as
// gt-vs-pred verdict cards when the columns are present.
const AXES = [
  { name: "nature", gt: "gt_nature", pred: "pred_nature" },
  { name: "biotic", gt: "gt_biotic", pred: "pred_biotic" },
  { name: "material", gt: "gt_material", pred: "pred_material" },
  { name: "image nature", gt: "image_gt_nature", pred: "image_pred_nature" },
];

// Columns offered as dropdown filters (when present + few distinct values).
const FILTER_CANDIDATES = [
  "dataset", "model", "class_name",
  "gt_nature", "pred_nature", "gt_biotic", "pred_biotic",
  "gt_material", "pred_material", "image_gt_nature", "image_pred_nature",
  "clipmatch_pred_nature", "clipmatch_pred_biotic", "clipmatch_pred_material",
  "clipmatch_top1_correct", "parse_failed",
];

// Columns handled specially, excluded from the generic key/value table.
const SPECIAL_COLS = new Set([
  "image_path", "caption", "reasoning", "objects", "gt_targets",
  ...AXES.flatMap(a => [a.gt, a.pred]),
]);

const TRUE_TOKENS = new Set(["true", "1", "yes", "biotic", "material", "y"]);
const FALSE_TOKENS = new Set(["false", "0", "no", "abiotic", "immaterial", "n"]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = sel => document.querySelector(sel);

function normBool(val) {
  // -> true / false / null (unknown / not applicable)
  if (val === null || val === undefined) return null;
  const s = String(val).trim().toLowerCase();
  if (s === "" || s === "none" || s === "nan" || s === "n/a") return null;
  if (TRUE_TOKENS.has(s)) return true;
  if (FALSE_TOKENS.has(s)) return false;
  return null;
}

function badge(rawVal) {
  const b = normBool(rawVal);
  const disp = (rawVal === "" || rawVal === null || rawVal === undefined) ? "n/a" : String(rawVal);
  const cls = b === true ? "t" : b === false ? "f" : "n";
  return `<span class="badge ${cls}">${escapeHtml(disp)}</span>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtVal(v) {
  if (v === "" || v === null || v === undefined) return '<span class="v-null">—</span>';
  const num = Number(v);
  if (!Number.isNaN(num) && /^-?\d*\.?\d+(e-?\d+)?$/i.test(String(v).trim())) {
    // Round floats for readability.
    if (String(v).includes(".")) return escapeHtml(num.toFixed(4).replace(/\.?0+$/, ""));
  }
  return escapeHtml(String(v));
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------
async function loadFiles() {
  const sel = $("#fileSelect");
  sel.innerHTML = "<option>Loading…</option>";
  const res = await fetch("/api/files");
  const data = await res.json();
  state.files = data.files || [];
  if (!state.files.length) {
    sel.innerHTML = "<option value=''>No prediction CSVs found</option>";
    return;
  }
  sel.innerHTML = state.files.map((f, i) => {
    const label = `${f.name}  ·  ${f.kind === "taxonomy_calibration" ? "taxonomy" : "pipeline"}` +
                  (f.dataset ? ` · ${f.dataset}` : "");
    return `<option value="${i}">${escapeHtml(label)}</option>`;
  }).join("");
  sel.value = "0";
  await loadData(state.files[0].path);
}

async function loadData(path) {
  $("#emptyState").textContent = "Loading…";
  $("#emptyState").classList.remove("hidden");
  $("#record").classList.add("hidden");
  const res = await fetch("/api/data?file=" + encodeURIComponent(path));
  const data = await res.json();
  if (data.error) {
    $("#emptyState").textContent = "Error: " + data.error;
    return;
  }
  state.file = path;
  state.columns = data.columns;
  state.kind = data.kind;
  state.rows = data.rows;
  state.filters = {};
  state.search = "";
  state.mismatchOnly = false;
  state.failOnly = false;
  $("#searchBox").value = "";
  buildFilters();
  applyFilters();
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------
function distinctValues(col) {
  const set = new Set();
  for (const r of state.rows) {
    const v = r[col];
    if (v !== undefined) set.add(v === null ? "" : String(v));
  }
  return [...set].sort();
}

function buildFilters() {
  const wrap = $("#filterControls");
  wrap.innerHTML = "";
  const colSet = new Set(state.columns);
  for (const col of FILTER_CANDIDATES) {
    if (!colSet.has(col)) continue;
    const vals = distinctValues(col);
    if (vals.length < 2 || vals.length > 25) continue;
    const row = document.createElement("div");
    row.className = "filter-row";
    const opts = ['<option value="">All</option>']
      .concat(vals.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v === "" ? "(empty)" : v)}</option>`))
      .join("");
    row.innerHTML = `<label>${escapeHtml(col)}</label><select data-col="${escapeHtml(col)}">${opts}</select>`;
    row.querySelector("select").addEventListener("change", e => {
      state.filters[col] = e.target.value;
      applyFilters();
    });
    wrap.appendChild(row);
  }

  // Toggles.
  const tog = $("#toggleControls");
  const hasAxis = AXES.some(a => colSet.has(a.gt) && colSet.has(a.pred));
  const hasFail = colSet.has("parse_failed") || colSet.has("parse_failure_count_image");
  tog.innerHTML =
    (hasAxis ? `<label><input type="checkbox" id="mismatchToggle"> Only gt ≠ pred mismatches</label>` : "") +
    (hasFail ? `<label><input type="checkbox" id="failToggle"> Only parse failures</label>` : "");
  if (hasAxis) $("#mismatchToggle").addEventListener("change", e => { state.mismatchOnly = e.target.checked; applyFilters(); });
  if (hasFail) $("#failToggle").addEventListener("change", e => { state.failOnly = e.target.checked; applyFilters(); });
}

function rowHasMismatch(r) {
  for (const a of AXES) {
    if (!(a.gt in r) || !(a.pred in r)) continue;
    const g = normBool(r[a.gt]), p = normBool(r[a.pred]);
    if (g !== null && p !== null && g !== p) return true;
  }
  return false;
}

function rowFailed(r) {
  if (normBool(r.parse_failed) === true) return true;
  const c = Number(r.parse_failure_count_image);
  if (!Number.isNaN(c) && c > 0) return true;
  // Any extracted object with a parse failure.
  if (Array.isArray(r._objects) && r._objects.some(o => o && o.parse_failed)) return true;
  return false;
}

function applyFilters() {
  const q = state.search.trim().toLowerCase();
  const view = [];
  state.rows.forEach((r, i) => {
    for (const [col, val] of Object.entries(state.filters)) {
      if (val === "") continue;
      if (String(r[col] ?? "") !== val) return;
    }
    if (state.mismatchOnly && !rowHasMismatch(r)) return;
    if (state.failOnly && !rowFailed(r)) return;
    if (q) {
      const hay = [
        r._image_name, r.class_name, r.caption, r.reasoning,
        r.clipmatch_pred_class, r.image_path,
      ].filter(Boolean).join(" ").toLowerCase();
      if (!hay.includes(q)) return;
    }
    view.push(i);
  });
  state.view = view;
  state.pos = 0;
  renderResultList();
  renderCounter();
  if (view.length) renderRecord(); else showEmpty("No rows match the current filters.");
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function showEmpty(msg) {
  $("#emptyState").textContent = msg;
  $("#emptyState").classList.remove("hidden");
  $("#record").classList.add("hidden");
}

function renderCounter() {
  $("#counter").textContent = `${state.view.length ? state.pos + 1 : 0} / ${state.view.length}`;
  $("#resultCount").textContent = `(${state.view.length} of ${state.rows.length})`;
}

function renderResultList() {
  const ul = $("#resultList");
  ul.innerHTML = "";
  const frag = document.createDocumentFragment();
  state.view.forEach((rowIdx, viewIdx) => {
    const r = state.rows[rowIdx];
    const li = document.createElement("li");
    if (viewIdx === state.pos) li.classList.add("active");
    let dotCls = "neutral";
    if (rowHasMismatch(r)) dotCls = "bad";
    else if (AXES.some(a => (a.gt in r) && (a.pred in r) && normBool(r[a.gt]) !== null)) dotCls = "good";
    const label = r._image_name + (r.class_name ? `  ·  ${r.class_name}` : "");
    li.innerHTML = `<span class="dot ${dotCls}"></span><span>${escapeHtml(label)}</span>`;
    li.addEventListener("click", () => { state.pos = viewIdx; renderRecord(); renderResultList(); renderCounter(); });
    frag.appendChild(li);
  });
  ul.appendChild(frag);
  const active = ul.querySelector("li.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function renderRecord() {
  if (!state.view.length) return;
  $("#emptyState").classList.add("hidden");
  $("#record").classList.remove("hidden");
  const r = state.rows[state.view[state.pos]];

  // --- image ---
  const wrap = $("#imageWrap");
  const dataset = r._dataset || "";
  if (r.image_path) {
    const src = "/api/image?dataset=" + encodeURIComponent(dataset) + "&path=" + encodeURIComponent(r.image_path);
    wrap.innerHTML = `<img src="${src}" alt="image" onerror="this.parentNode.innerHTML='<div class=\\'missing\\'>Image not found locally.<br>Check DATASET_IMAGE_ROOTS in config.py</div>'">`;
  } else {
    wrap.innerHTML = `<div class="missing">No image_path in this row.</div>`;
  }
  $("#imagePath").textContent = (dataset ? `[${dataset}] ` : "") + (r.image_path || "");

  // --- verdicts ---
  const vparts = [];
  for (const a of AXES) {
    if (!(a.gt in r) && !(a.pred in r)) continue;
    const g = normBool(r[a.gt]), p = normBool(r[a.pred]);
    const known = g !== null && p !== null;
    const cls = !known ? "" : (g === p ? "match" : "mismatch");
    const tag = !known ? "" : (g === p ? `<span class="tag-match">✓ match</span>` : `<span class="tag-mismatch">✗ mismatch</span>`);
    vparts.push(`<div class="verdict ${cls}">
        <div class="axis">${a.name}</div>
        <div class="pair">${badge(r[a.gt])}<span class="arrow">→</span>${badge(r[a.pred])} ${tag}</div>
      </div>`);
  }
  $("#verdicts").innerHTML = vparts.join("");

  // --- details ---
  const blocks = [];
  if (r.caption) blocks.push(`<div class="block"><h3>Caption</h3><div class="caption-text">${escapeHtml(r.caption)}</div></div>`);
  if (r.reasoning) blocks.push(`<div class="block"><h3>Reasoning</h3><div class="reasoning-text">${escapeHtml(r.reasoning)}</div></div>`);
  if (Array.isArray(r._objects) && r._objects.length) blocks.push(renderObjects(r._objects));
  if (r._gt_targets) blocks.push(renderTargets(r._gt_targets));
  blocks.push(renderKV(r));
  $("#details").innerHTML = blocks.join("");
}

function renderObjects(objs) {
  const rows = objs.map(o => {
    const src = k => o[k] ? `<span class="pill ${o[k] === "map" || o[k] === "mapping" ? "map" : "vlm"}">${escapeHtml(o[k])}</span>` : "—";
    const tri = v => {
      const b = normBool(v);
      const c = b === true ? "v-true" : b === false ? "v-false" : "v-null";
      return `<span class="${c}">${v === null || v === undefined || v === "" ? "—" : escapeHtml(String(v))}</span>`;
    };
    return `<tr>
      <td>${escapeHtml(o.text ?? "")}</td>
      <td>${o.mapped ? `<span class="pill map">${escapeHtml(o.mapped_synset || "mapped")}</span>` : `<span class="pill vlm">vlm</span>`}</td>
      <td>${tri(o.nature)}</td>
      <td>${tri(o.biotic)}</td>
      <td>${tri(o.material)}</td>
      <td>${src("nature_source")}</td>
      <td>${o.parse_failed ? '<span class="v-false">yes</span>' : "—"}</td>
    </tr>`;
  }).join("");
  return `<div class="block"><h3>Extracted objects (${objs.length})</h3>
    <table class="objs"><thead><tr>
      <th>object</th><th>mapped</th><th>nature</th><th>biotic</th><th>material</th><th>src</th><th>parse fail</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderTargets(gt) {
  const targets = (gt && gt.targets) || [];
  const matches = (gt && gt.target_matches) || [];
  let html = `<div class="block"><h3>Ground-truth targets</h3>`;
  if (targets.length) {
    html += `<table class="objs"><thead><tr><th>class</th><th>synset</th><th>nature</th><th>biotic</th><th>material</th></tr></thead><tbody>`;
    html += targets.map(t => `<tr>
        <td>${escapeHtml(t.class_name ?? "")}</td>
        <td>${escapeHtml(t.synset_id ?? "—")}</td>
        <td>${badge(t.gt_nature)}</td><td>${badge(t.gt_biotic)}</td><td>${badge(t.gt_material)}</td>
      </tr>`).join("");
    html += `</tbody></table>`;
  } else {
    html += `<div class="muted">No structured targets.</div>`;
  }
  if (matches.length) {
    html += `<h3 style="margin-top:14px">Matched objects per target</h3>`;
    html += `<table class="objs"><thead><tr><th>class</th><th>matched object</th><th>gt→pred biotic</th><th>gt→pred material</th></tr></thead><tbody>`;
    html += matches.map(m => `<tr>
        <td>${escapeHtml(m.class_name ?? "")}</td>
        <td>${m.matched_object_text ? escapeHtml(m.matched_object_text) : '<span class="v-null">no match</span>'}</td>
        <td>${"gt_biotic" in m ? badge(m.gt_biotic) + " → " + badge(m.pred_biotic) : "—"}</td>
        <td>${"gt_material" in m ? badge(m.gt_material) + " → " + badge(m.pred_material) : "—"}</td>
      </tr>`).join("");
    html += `</tbody></table>`;
  }
  return html + `</div>`;
}

function renderKV(r) {
  const rows = [];
  for (const col of state.columns) {
    if (SPECIAL_COLS.has(col)) continue;
    if (col === "dataset") continue; // shown under the image
    rows.push(`<tr><td class="k">${escapeHtml(col)}</td><td class="v">${fmtVal(r[col])}</td></tr>`);
  }
  if (!rows.length) return "";
  return `<div class="block"><h3>Fields</h3><table class="kv"><tbody>${rows.join("")}</tbody></table></div>`;
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function go(delta) {
  if (!state.view.length) return;
  state.pos = (state.pos + delta + state.view.length) % state.view.length;
  renderRecord(); renderResultList(); renderCounter();
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function init() {
  $("#fileSelect").addEventListener("change", e => {
    const f = state.files[Number(e.target.value)];
    if (f) loadData(f.path);
  });
  $("#reloadFiles").addEventListener("click", loadFiles);
  $("#prevBtn").addEventListener("click", () => go(-1));
  $("#nextBtn").addEventListener("click", () => go(1));
  $("#jumpInput").addEventListener("change", e => {
    const n = parseInt(e.target.value, 10);
    if (!Number.isNaN(n) && n >= 1 && n <= state.view.length) {
      state.pos = n - 1; renderRecord(); renderResultList(); renderCounter();
    }
  });
  $("#searchBox").addEventListener("input", e => { state.search = e.target.value; applyFilters(); });
  $("#resetFilters").addEventListener("click", () => {
    state.filters = {}; state.search = ""; state.mismatchOnly = false; state.failOnly = false;
    buildFilters(); $("#searchBox").value = ""; applyFilters();
  });
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") go(-1);
    if (e.key === "ArrowRight") go(1);
  });
  loadFiles();
}

init();
