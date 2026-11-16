"""Write a self-contained index.html for a comparison run (data embedded inline, images by relative path)."""

import html
import json
import pathlib

_CSS = """
:root {
  --blue-70: #1a4480; --blue-60: #005ea2;
  --surface-0: #f0f0f0; --surface-1: #ffffff; --border: #dfe1e2;
  --text-primary: #1b1b1b; --text-secondary: #565c65; --text-muted: #71767a;
  --error: #b50909;
  --font-sans: "Public Sans", system-ui, -apple-system, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --error: #e66767;
  }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font-sans); color: var(--text-primary); background: var(--surface-0); line-height: 1.5; }
header { background: var(--blue-70); color: #fff; padding: 1.75rem 1.5rem; }
header .agency { font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; opacity: .75; }
header h1 { font-size: 1.7rem; }
header .meta { opacity: .85; font-size: .9rem; margin-top: .25rem; }
nav { background: var(--blue-60); padding: 0 1.5rem; display: flex; flex-wrap: wrap; }
nav a { color: rgba(255,255,255,.85); text-decoration: none; padding: .6rem 1rem; font-size: .875rem; font-weight: 600; }
nav a:hover { color: #fff; background: rgba(255,255,255,.08); }
main { max-width: 1400px; margin: 0 auto; padding: 1.5rem; }
section { background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; }
h2 { font-size: 1.25rem; margin-bottom: .25rem; }
h3 { font-size: 1.05rem; margin: 1.25rem 0 .5rem; }
p.note { color: var(--text-secondary); font-size: .9rem; margin-bottom: .75rem; }
.caveat { border-left: 4px solid #e5a000; background: rgba(229,160,0,.1); padding: .55rem .8rem; border-radius: 4px;
          font-size: .9rem; margin-bottom: .9rem; }
table { border-collapse: collapse; font-size: .875rem; font-variant-numeric: tabular-nums; }
th, td { padding: .4rem .7rem; border-bottom: 1px solid var(--border); text-align: right; }
th { color: var(--text-secondary); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
td.err { color: var(--error); }
.scroll { overflow-x: auto; }
.controls { display: flex; gap: .75rem; align-items: center; margin-bottom: .75rem; font-size: .9rem; }
select { font: inherit; padding: .25rem .5rem; background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border); border-radius: 4px; }
/* heatmap */
table.heat td.cell { text-align: center; min-width: 4.5rem; border: 2px solid var(--surface-1); cursor: default; }
table.heat td.diag { color: var(--text-muted); text-align: center; }
table.heat th.col { writing-mode: vertical-rl; transform: rotate(180deg); text-align: left; padding: .5rem .3rem; }
.legend { display: flex; align-items: center; gap: .5rem; font-size: .8rem; color: var(--text-secondary); margin-top: .6rem; }
.legend .ramp { display: flex; } .legend .ramp span { width: 22px; height: 12px; }
#tip { position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border);
       border-radius: 4px; padding: .4rem .6rem; font-size: .8rem; box-shadow: 0 2px 8px rgba(0,0,0,.18); display: none; z-index: 10; }
/* gallery */
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: .9rem; }
figure { margin: 0; }
figure img { width: 100%; display: block; border-radius: 4px; background: #000; }
figcaption { font-size: .8rem; color: var(--text-secondary); margin-top: .25rem; display: flex; justify-content: space-between; }
figcaption b { color: var(--text-primary); }
figcaption .err { color: var(--error); }
details { margin-top: .5rem; } summary { cursor: pointer; color: var(--text-secondary); font-size: .9rem; }
table.phase td, table.phase th { text-align: left; vertical-align: top; }
table.phase td.num { text-align: right; }
td.score { text-align: center; }
.sub { color: var(--text-muted); font-size: .8rem; }
/* viewers */
.vbar { display: flex; flex-wrap: wrap; gap: .6rem 1.2rem; align-items: center; margin-bottom: .9rem; font-size: .9rem; }
.vbar label { color: var(--text-secondary); }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.seg button { font: inherit; font-size: .85rem; padding: .3rem .75rem; border: 0; background: var(--surface-1);
              color: var(--text-primary); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--blue-60); color: #fff; }
.navbtn { font: inherit; padding: .25rem .6rem; border: 1px solid var(--border); border-radius: 4px;
          background: var(--surface-1); color: var(--text-primary); cursor: pointer; }
.kbd { color: var(--text-muted); font-size: .8rem; }
.panes { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.pane .phead { display: flex; gap: .5rem; align-items: center; margin-bottom: .4rem; }
.pane img { width: 100%; display: block; border-radius: 4px; background: #000; image-rendering: pixelated; }
.imgwrap { position: relative; }
.imgwrap svg { position: absolute; inset: 0; width: 100%; height: 100%; }
/* boxes are SVG vectors over the plain image (not baked into the PNG) so they stay crisp at any display size */
.imgwrap rect { fill: transparent; stroke: var(--c); stroke-width: 1.5px; vector-effect: non-scaling-stroke; cursor: crosshair; }
.imgwrap rect:hover, .imgwrap rect.hl { stroke: #ff3df2; stroke-width: 3px; fill: rgba(255,61,242,.18); }
.imgwrap rect.hl-weak { stroke: #ff3df2; stroke-width: 2.5px; stroke-dasharray: 4 3; }
.mgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: .5rem; }
.mgrid figure { cursor: grab; border: 2px dashed transparent; border-radius: 6px; padding: 2px; }
.mgrid figure.dragging { opacity: .4; }
.mgrid figure.drop { border-color: var(--blue-60); }
.mgrid figure.on { border-style: solid; border-color: var(--blue-60); }
.mgrid figure img { image-rendering: pixelated; }
.mgrid figcaption { font-size: .9rem; margin-top: .35rem; }
@media (max-width: 900px) { .mgrid { grid-template-columns: repeat(2, 1fr); } }
#tip { max-width: 380px; }
#tip .sub { display: block; }
.pane .cap { font-size: .85rem; color: var(--text-secondary); margin-top: .35rem; }
.pane .cap b { color: var(--text-primary); }
.pane .cap .err { color: var(--error); }
.regions { max-height: 320px; overflow: auto; margin-top: .6rem; }
@media (max-width: 900px) { .panes { grid-template-columns: 1fr; } }
ul.mlist { margin: 0; padding-left: 1rem; font-size: .85rem; }
pre { white-space: pre-wrap; font-size: .8rem; background: var(--surface-0); padding: .75rem; border-radius: 4px; margin-top: .5rem; }
"""

# Sequential blue ramp (100 → 700), light → dark
_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

_JS = r"""
const DATA = JSON.parse(document.getElementById('data').textContent);
const RAMP = %RAMP%;
const tip = document.getElementById('tip');
function label(m) { return m.replace('opencv:', 'OpenCV ').replace('vlm:', 'VLM ').replace('stardist', 'StarDist').replace(/:/g, ' · '); }
function rampColor(v) { return RAMP[Math.min(RAMP.length - 1, Math.floor(v * RAMP.length))]; }
function inkFor(v) { return Math.floor(v * RAMP.length) >= 7 ? '#ffffff' : '#1b1b1b'; }

function agreementFor(key) {
  if (key === '__all__') return DATA.overall_agreement;
  const [s, e] = key.split('/');
  return DATA.samples[s].elements[e].agreement;
}
function renderHeat(key) {
  const ag = agreementFor(key), M = DATA.methods;
  let h = '<table class="heat"><tr><th></th>' + M.map(m => `<th class="col">${label(m)}</th>`).join('') + '</tr>';
  M.forEach((a, i) => {
    h += `<tr><th>${label(a)}</th>`;
    M.forEach((b, j) => {
      if (i === j) { h += '<td class="diag">—</td>'; return; }
      const r = ag[i < j ? `${a}|${b}` : `${b}|${a}`];
      h += `<td class="cell" style="background:${rampColor(r.f1)};color:${inkFor(r.f1)}"
              data-a="${label(a)}" data-b="${label(b)}" data-f1="${r.f1.toFixed(2)}"
              data-m="${r.matched}" data-iou="${r.mean_iou.toFixed(2)}">${r.f1.toFixed(2)}</td>`;
    });
    h += '</tr>';
  });
  document.getElementById('heat').innerHTML = h + '</table>';
}
document.getElementById('heat').addEventListener('mousemove', ev => {
  const c = ev.target.closest('td.cell');
  if (!c) { tip.style.display = 'none'; return; }
  tip.innerHTML = `<b>${c.dataset.a}</b> vs <b>${c.dataset.b}</b><br>F1 ${c.dataset.f1} · ${c.dataset.m} matched boxes<br>mean IoU of matches ${c.dataset.iou}`;
  tip.style.display = 'block';
  tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 14) + 'px';
});
document.getElementById('heat').addEventListener('mouseleave', () => tip.style.display = 'none');
const sel = document.getElementById('heat-sel');
sel.addEventListener('change', () => renderHeat(sel.value));
renderHeat('__all__');

/* ---------- box hover overlay (boxes are fractional 0–1; shape = [h, w] of the TIFF) ---------- */
function px(b, shape) {
  const [h, w] = shape;
  return { x1: b.x1 * w, x2: b.x2 * w, y1: b.y1 * h, y2: b.y2 * h };
}
function locHTML(b, shape) {
  const p = px(b, shape), f = v => v.toFixed(0);
  return `x ${f(p.x1)}–${f(p.x2)}, y ${f(p.y1)}–${f(p.y2)} px` +
    `<span class="sub">center (${f((p.x1 + p.x2) / 2)}, ${f((p.y1 + p.y2) / 2)}) · ${f(p.x2 - p.x1)}×${f(p.y2 - p.y1)} px` +
    ` · image ${shape[1]}×${shape[0]}</span>`;
}
function inter(a, b) {
  return Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)) * Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
}
const area = b => (b.x2 - b.x1) * (b.y2 - b.y1);
const iou = (a, b) => { const i = inter(a, b); return i ? i / (area(a) + area(b) - i) : 0; };
const ovl = (a, b) => { const i = inter(a, b); return i ? i / Math.min(area(a), area(b)) : 0; };
function bestMatch(b, others, fn) {
  let best = null;
  (others || []).forEach((o, k) => { const s = fn(b, o); if (s > 0 && (!best || s > best.s)) best = { k, s }; });
  return best;
}
/* draw invisible hover rects; smaller boxes last so they sit on top of the ones containing them */
const DET_COLOR = '#eda100', MAT_COLOR = '#00e5ff';  // same colours the saved *_boxes.png files use
function drawOverlay(svg, boxes, on, color) {
  svg.style.setProperty('--c', color);
  svg.innerHTML = on ? (boxes || []).map((b, k) => [b, k]).sort((p, q) => area(q[0]) - area(p[0]))
    .map(([b, k]) => `<rect data-k="${k}" x="${b.x1}" y="${b.y1}" width="${b.x2 - b.x1}" height="${b.y2 - b.y1}"/>`).join('') : '';
}
function showTip(ev, h) {
  if (!st.tips) { tip.style.display = 'none'; return; }
  tip.innerHTML = h; tip.style.display = 'block';
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX + 14, innerWidth - r.width - 8) + 'px';
  tip.style.top = Math.min(ev.clientY + 14, innerHeight - r.height - 8) + 'px';
}
/* hover on a pane's box → tooltip; also outline the best-matching box in the other pane */
function hookOverlay(prefix, describe) {
  for (const [k, o] of [['a', 'b'], ['b', 'a']]) {
    const svg = $(prefix + '-svg-' + k);
    svg.addEventListener('mousemove', ev => {
      const r = ev.target.closest('rect');
      document.querySelectorAll(`#${prefix}-svg-${o} rect.hl`).forEach(x => x.classList.remove('hl'));
      if (!r) { tip.style.display = 'none'; return; }
      const { html, match } = describe(k, o, +r.dataset.k);
      if (match != null) { const m = document.querySelector(`#${prefix}-svg-${o} rect[data-k="${match}"]`); if (m) m.classList.add('hl'); }
      showTip(ev, html);
    });
    svg.addEventListener('mouseleave', () => {
      tip.style.display = 'none';
      document.querySelectorAll(`#${prefix}-svg-${o} rect.hl`).forEach(x => x.classList.remove('hl'));
    });
  }
}

/* ---------- "all methods" grid: 4 tiles per row, linked hover, drag to reorder ----------
   g = { el, order: [...methods], tile(m) → {src, caption}, boxes(m) → [box], head(m, i) → html,
         score: iou | ovl, thresh, line(m, i, o, best) → html, rerender(), click(m)? } */
function makeGrid(g) {
  g.render = () => {
    g.el.innerHTML = g.order.map(m => {
      const t = g.tile(m);
      return `<figure draggable="true" data-m="${m}"><div class="imgwrap"><img draggable="false" loading="lazy" src="${t.src || ''}" alt="${label(m)}">
        <svg data-m="${m}" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
        <figcaption><b>${label(m)}</b>${t.caption}</figcaption></figure>`;
    }).join('');
    g.el.querySelectorAll('svg').forEach(svg => drawOverlay(svg, g.boxes(svg.dataset.m), st.boxes, g.color));
  };
  const clearHL = () => g.el.querySelectorAll('rect.hl, rect.hl-weak').forEach(x => x.classList.remove('hl', 'hl-weak'));
  g.el.addEventListener('mousemove', ev => {
    const rect = ev.target.closest('rect');
    clearHL();
    if (!rect) { tip.style.display = 'none'; return; }
    const m = rect.closest('svg').dataset.m, i = +rect.dataset.k, b = g.boxes(m)[i];
    const lines = M.filter(o => o !== m).map(o => {
      const best = bestMatch(b, g.boxes(o), g.score);
      if (best) {
        const el = g.el.querySelector(`svg[data-m="${CSS.escape(o)}"] rect[data-k="${best.k}"]`);
        if (el) el.classList.add(best.s >= g.thresh ? 'hl' : 'hl-weak');
      }
      return `<span class="sub">${!best ? '—' : best.s >= g.thresh ? '✓' : '~'} ${label(o)}: ${best ? g.line(m, i, o, best) : 'nothing here'}</span>`;
    });
    showTip(ev, g.head(m, i) + `<br><b>Same spot in other methods</b> (✓ matched · ~ overlaps · — none)` + lines.join(''));
  });
  g.el.addEventListener('mouseleave', () => { tip.style.display = 'none'; clearHL(); });
  if (g.click) g.el.addEventListener('click', ev => { const f = ev.target.closest('figure'); if (f) g.click(f.dataset.m); });
  /* drag a tile onto another to move it there */
  let dragM = null;
  const clearDrop = () => g.el.querySelectorAll('figure.drop').forEach(f => f.classList.remove('drop'));
  g.el.addEventListener('dragstart', ev => {
    const f = ev.target.closest('figure'); if (!f) return;
    dragM = f.dataset.m; f.classList.add('dragging'); tip.style.display = 'none';
    ev.dataTransfer.effectAllowed = 'move'; ev.dataTransfer.setData('text/plain', dragM);
  });
  g.el.addEventListener('dragover', ev => {
    const f = ev.target.closest('figure'); if (!f || !dragM) return;
    ev.preventDefault(); clearDrop(); if (f.dataset.m !== dragM) f.classList.add('drop');
  });
  g.el.addEventListener('drop', ev => {
    ev.preventDefault();
    const f = ev.target.closest('figure');
    if (f && dragM && f.dataset.m !== dragM) {
      const to = g.order.indexOf(f.dataset.m);
      g.order.splice(g.order.indexOf(dragM), 1);
      g.order.splice(to, 0, dragM);
      g.rerender();
    }
  });
  g.el.addEventListener('dragend', () => { dragM = null; clearDrop();
    g.el.querySelectorAll('figure.dragging').forEach(f => f.classList.remove('dragging')); });
  return g;
}

/* ---------- shared view controls: every .seg[data-ctl] on the page shows and sets the same state ---------- */
const V = DATA.viewer, M = DATA.methods;
const st = { idx: 0, boxes: true, view: 'original', tips: true, a: M[0], b: DATA.vlm[0] || M[1] || M[0] };
const $ = id => document.getElementById(id);
function syncControls() {
  const cur = { boxes: st.boxes ? 'on' : 'off', view: st.view, tips: st.tips ? 'on' : 'off' };
  document.querySelectorAll('.seg[data-ctl] button').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.v === cur[b.parentNode.dataset.ctl])));
}
function renderAll() { renderViewer(); renderMat(); syncControls(); }
document.querySelectorAll('.seg[data-ctl] button').forEach(b => b.onclick = () => {
  const c = b.parentNode.dataset.ctl, v = b.dataset.v;
  if (c === 'boxes') st.boxes = v === 'on';
  else if (c === 'view') st.view = v;
  else if (c === 'tips') { st.tips = v === 'on'; tip.style.display = 'none'; }
  renderAll();
});

/* ---------- detection viewer ---------- */
function fillSelect(sel, val) { sel.innerHTML = M.map(m => `<option value="${m}"${m === val ? ' selected' : ''}>${label(m)}</option>`).join(''); }
function pick(imgs, view, boxes) {
  const mode = (view === 'input' && imgs.input) ? 'input' : 'original';
  return { src: imgs[mode], mode };  // boxes are drawn as SVG on top, not taken from the *_boxes.png
}
function paneHTML(m, item) {
  const imgs = (V.images[item.key] || {})[m] || {};
  const { src, mode } = pick(imgs, st.view, st.boxes);
  const n = (V.counts[item.key] || {})[m], err = (V.errors[item.key] || {})[m];
  let what = mode === 'input' ? 'model input: ' + (DATA.model_input[m] || '?') : 'original normalized image';
  if (st.view === 'input' && mode !== 'input') what += ' (no model input saved — method errored?)';
  return [src, `<b>${label(m)}</b> · ${err ? `<span class="err" title="${err.replace(/"/g, '&quot;')}">error</span>` : `${n} boxes`}`
               + ` · ${what}${st.boxes ? ', with boxes' : ', no boxes'}`];
}
const vItem = () => V.items[st.idx];
const vBoxes = m => (V.boxes[vItem().key] || {})[m] || [];
const vGrid = makeGrid({
  el: $('v-grid'), order: [...M], color: DET_COLOR, score: iou, thresh: DATA.match_iou,
  rerender: () => renderViewer(),
  tile: m => {
    const err = (V.errors[vItem().key] || {})[m], src = pick((V.images[vItem().key] || {})[m] || {}, st.view, st.boxes).src;
    return { src, caption: err ? '<span class="err">error</span>' : `<span>${vBoxes(m).length} boxes</span>` };
  },
  boxes: vBoxes,
  head: (m, i) => `<b>${label(m)}</b> · box #${i + 1} of ${vBoxes(m).length} · ${vItem().label}<br>${locHTML(vBoxes(m)[i], V.shape[vItem().key])}`,
  line: (m, i, o, best) => `box #${best.k + 1}, IoU ${best.s.toFixed(2)}`,
  click: m => { st.a = m; $('v-sel-a').value = m; renderViewer(); },
});
function renderViewer() {
  const item = vItem();
  $('v-el').value = String(st.idx);
  for (const k of ['a', 'b']) {
    const [src, cap] = paneHTML(st[k], item);
    $('v-img-' + k).src = src || ''; $('v-img-' + k).alt = cap.replace(/<[^>]+>/g, '');
    $('v-cap-' + k).innerHTML = cap;
    drawOverlay($('v-svg-' + k), vBoxes(st[k]), st.boxes, DET_COLOR);
  }
  vGrid.render();
  vGrid.el.querySelectorAll('figure').forEach(f => f.classList.toggle('on', f.dataset.m === st.a));
}
$('v-el').innerHTML = V.items.map((it, i) => `<option value="${i}">${it.label}</option>`).join('');
fillSelect($('v-sel-a'), st.a); fillSelect($('v-sel-b'), st.b);
$('v-el').onchange = e => { st.idx = +e.target.value; renderViewer(); };
$('v-prev').onclick = () => { st.idx = (st.idx - 1 + V.items.length) % V.items.length; renderViewer(); };
$('v-next').onclick = () => { st.idx = (st.idx + 1) % V.items.length; renderViewer(); };
$('v-sel-a').onchange = e => { st.a = e.target.value; renderViewer(); };
$('v-sel-b').onchange = e => { st.b = e.target.value; renderViewer(); };
$('v-grid-reset').onclick = () => { vGrid.order = [...M]; renderViewer(); };
document.addEventListener('keydown', e => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowRight') $('v-next').click();
  else if (e.key === 'ArrowLeft') $('v-prev').click();
  else if (e.key === 'b') { st.boxes = !st.boxes; renderAll(); }
  else if (e.key === 'i') { st.view = st.view === 'input' ? 'original' : 'input'; renderAll(); }
  else if (e.key === 't') { st.tips = !st.tips; tip.style.display = 'none'; syncControls(); }
});
hookOverlay('v', (k, o, i) => {
  const item = vItem(), shape = V.shape[item.key];
  const b = vBoxes(st[k])[i], m = bestMatch(b, vBoxes(st[o]), iou);
  const other = `<b>${label(st[o])}</b>`;
  return {
    match: m && m.k,
    html: `<b>${label(st[k])}</b> · box #${i + 1} of ${vBoxes(st[k]).length} · ${item.label}<br>${locHTML(b, shape)}` +
      (m ? `<span class="sub">Best overlap in ${other}: box #${m.k + 1}, IoU ${m.s.toFixed(2)}` +
           `${m.s >= DATA.match_iou ? ' (counts as matched)' : ` (below ${DATA.match_iou}, not matched)`}</span>` +
           `<span class="sub">${other} box: ${locHTML(vBoxes(st[o])[m.k], shape).replace(/<[^>]+>/g, ' ')}</span>`
         : `<span class="sub">No overlapping box in ${other}</span>`)
  };
});

/* ---------- material-group filter ---------- */
document.querySelectorAll('.grp-filter').forEach(sel => sel.onchange = () => {
  document.querySelectorAll('#' + CSS.escape(sel.dataset.target) + ' tr[data-a]').forEach(tr =>
    tr.style.display = (!sel.value || tr.dataset.a === sel.value) ? '' : 'none');
});

/* ---------- materials viewer ---------- */
const MAT = DATA.materials, MS = Object.keys(MAT);
const mt = { s: MS[0], a: M[0], b: DATA.vlm[0] || M[1] || M[0] };
const mRegs = m => (MAT[mt.s].regions[m] || []);
function regTable(regs, shape) {
  if (!regs || !regs.length) return '<p class="note">No multi-element regions.</p>';
  return '<table class="phase"><tr><th>#</th><th>Location</th><th>Elements</th><th>Material</th><th>Confidence</th></tr>' +
    regs.map(r => `<tr><td>${r.id}</td><td>${locHTML(r.box, shape)}</td><td>${r.elements.join(', ')}</td><td>${r.material}${r.alts ? `<div class="sub">Alternatives: ${r.alts}</div>` : ''}</td><td>${r.confidence}</td></tr>`).join('') + '</table>';
}
/* material name with what it is: an LLM guess (self-rated confidence) or a library ratio match */
function matText(m, r) {
  return m.startsWith('vlm:') ? `LLM best guess: ${r.material} (self-rated ${r.confidence})`
       : r.material === 'no library match' ? r.material : `library match: ${r.material} (${r.confidence})`;
}
function ratiosText(r) {
  return Object.entries(r).sort((p, q) => q[1] - p[1]).map(([el, v]) => `${el} ${v}%`).join(', ');
}
const mGrid = MS.length && makeGrid({
  el: $('m-grid'), order: [...M], color: MAT_COLOR, score: ovl, thresh: DATA.phase_match_overlap,
  rerender: () => renderMat(),
  tile: m => {
    const d = MAT[mt.s], imgs = d.images[m] || {}, err = d.errors[m];
    return { src: imgs.plain,
             caption: err ? '<span class="err">error</span>' : `<span>${mRegs(m).length} regions</span>` };
  },
  boxes: m => mRegs(m).map(r => r.box),
  head: (m, i) => {
    const r = mRegs(m)[i];
    return `<b>${label(m)}</b> · region ${r.id}<br>${locHTML(r.box, MAT[mt.s].shape)}` +
      `<span class="sub"><b>${r.elements.join(', ')}</b> → ${matText(m, r)}</span>` +
      (Object.keys(r.ratios).length ? `<span class="sub">Ratios: ${ratiosText(r.ratios)}</span>` : '') +
      (r.reasoning ? `<span class="sub">LLM: ${r.reasoning}</span>` : '') +
      (r.alts ? `<span class="sub">Alternatives: ${r.alts}</span>` : '');
  },
  line: (m, i, o, best) => { const mr = mRegs(o)[best.k]; return `${mr.elements.join(', ')} → ${mr.material} (overlap ${best.s.toFixed(2)})`; },
});
function renderMat() {
  if (!MS.length) return;
  const d = MAT[mt.s];
  for (const k of ['a', 'b']) {
    const m = mt[k], imgs = d.images[m] || {}, regs = mRegs(m), err = d.errors[m];
    $('m-img-' + k).src = imgs.plain || '';
    $('m-cap-' + k).innerHTML = `<b>${label(m)}</b> · ${err ? '<span class="err">error</span>' : regs.length + ' material regions'} · ${st.boxes ? 'with boxes' : 'no boxes'}`;
    $('m-reg-' + k).innerHTML = regTable(regs, d.shape);
    drawOverlay($('m-svg-' + k), regs.map(r => r.box), st.boxes, MAT_COLOR);
  }
  mGrid.render();
}
if (MS.length) {
  $('m-sample').innerHTML = MS.map(s => `<option>${s}</option>`).join('');
  $('m-sample').onchange = e => { mt.s = e.target.value; renderMat(); };
  fillSelect($('m-sel-a'), mt.a); fillSelect($('m-sel-b'), mt.b);
  $('m-sel-a').onchange = e => { mt.a = e.target.value; renderMat(); };
  $('m-sel-b').onchange = e => { mt.b = e.target.value; renderMat(); };
  $('m-grid-reset').onclick = () => { mGrid.order = [...M]; renderMat(); };
  hookOverlay('m', (k, o, i) => {
    const d = MAT[mt.s], r = mRegs(mt[k])[i];
    const m = bestMatch(r.box, mRegs(mt[o]).map(x => x.box), ovl);
    const other = `<b>${label(mt[o])}</b>`, mr = m && mRegs(mt[o])[m.k];
    return {
      match: m && m.k,
      html: `<b>${label(mt[k])}</b> · region ${r.id}<br>${locHTML(r.box, d.shape)}` +
        `<span class="sub"><b>${r.elements.join(', ')}</b> → ${matText(mt[k], r)}</span>` +
        (Object.keys(r.ratios).length ? `<span class="sub">Ratios: ${ratiosText(r.ratios)}</span>` : '') +
        (r.reasoning ? `<span class="sub">LLM: ${r.reasoning}</span>` : '') +
        (r.alts ? `<span class="sub">Alternatives: ${r.alts}</span>` : '') +
        (m ? `<span class="sub">Best overlap in ${other}: region ${mr.id} (${mr.elements.join(', ')} → ${mr.material}), ` +
             `overlap ${m.s.toFixed(2)}${m.s >= DATA.phase_match_overlap ? ' (counts as matched)' : ' (not matched)'}</span>`
           : `<span class="sub">No overlapping region in ${other}</span>`)
    };
  });
}
renderAll();
"""




def _label(m: str) -> str:
    return (m.replace("opencv:", "OpenCV ").replace("vlm:", "VLM ").replace("stardist", "StarDist")
             .replace(":", " · "))


def _ramp(v: float) -> tuple:
    i = min(len(_RAMP) - 1, int(v * len(_RAMP)))
    return _RAMP[i], ("#ffffff" if i >= 7 else "#1b1b1b")


def _sample_shape(sd: dict) -> list:
    """[h, w] of the sample's TIFFs (every element is resized to AutoMap's common shape)."""
    return next(iter(sd["elements"].values()), {}).get("shape") or [1, 1]


def _loc(b: dict, shape: list) -> str:
    """Pixel location of a fractional box on the TIFF: x range, y range, center."""
    h, w = shape
    x1, x2, y1, y2 = b["x1"] * w, b["x2"] * w, b["y1"] * h, b["y2"] * h
    return (f"x {x1:.0f}–{x2:.0f}, y {y1:.0f}–{y2:.0f}"
            f"<div class='sub'>center ({(x1 + x2) / 2:.0f}, {(y1 + y2) / 2:.0f}) · "
            f"{x2 - x1:.0f}×{y2 - y1:.0f} px</div>")


def _phase_section(results: dict) -> str:
    e = html.escape
    am, vv = results.get("automap_methods", []), results.get("vlm_variants", [])
    out = []
    for s, sd in results["samples"].items():
        ph = sd["phase"]
        shape = _sample_shape(sd)
        out.append(f"<h3>{e(s)}</h3>")
        # summary grid: AutoMap method × VLM variant
        if am and vv and ph["comparisons"]:
            head = "<tr><th>AutoMap method</th>" + "".join(f"<th>{e(_label(v))}</th>" for v in vv) + "</tr>"
            rows = []
            for a in am:
                cells = []
                for v in vv:
                    c = ph["comparisons"].get(f"{a}|{v}")
                    if not c:
                        cells.append('<td class="err">error</td>')
                        continue
                    bg, ink = _ramp(c["f1"])
                    cells.append(
                        f'<td class="score" style="background:{bg};color:{ink}" '
                        f'title="{c["matched"]} matched of {c["n_automap"]} AutoMap unions / {c["n_vlm"]} LLM zones; '
                        f'mean element Jaccard {c["mean_element_jaccard"]:.2f}">'
                        f'F1 {c["f1"]:.2f}<br><span style="font-size:.75rem">{c["matched"]} of '
                        f'{c["n_automap"]} / {c["n_vlm"]} · elem {c["mean_element_jaccard"]:.2f}</span></td>')
                rows.append(f"<tr><th>{e(_label(a))}</th>{''.join(cells)}</tr>")
            out.append('<div class="scroll"><table class="heat">' + head + "".join(rows) + "</table></div>")
            out.append('<p class="note" style="margin-top:.5rem">Each cell: region F1, then matched regions of '
                       '(AutoMap multi-element unions / LLM zones), then mean element-set Jaccard of matched pairs '
                       '(1.00 = same elements). Hover for details.</p>')
        # counts + time per method
        rows = []
        for mid in results["methods"]:
            n = len(ph["regions"].get(mid, [])) if mid in ph["regions"] else None
            t = ph["times"].get(mid)
            err = ph["errors"].get(mid)
            flag = ph.get("flags", {}).get(mid)
            cell = (f'<td class="err" title="{e(err)}">error</td>' if err else
                    f'<td class="num err" title="{e(flag)}">⚠ {n}</td>' if flag else
                    f'<td class="num">{n if n is not None else "—"}</td>')
            rows.append(f"<tr><td>{e(_label(mid))}</td>" + cell
                        + f'<td class="num">{f"{t:.1f}" if t is not None else "—"}</td></tr>')
        out.append('<div class="scroll"><table class="phase" style="margin-top:.75rem"><tr><th>Method</th>'
                   '<th>Multi-element regions</th><th>Time (s)</th></tr>' + "".join(rows) + "</table></div>")
        # side-by-side matched regions
        for key, c in ph["comparisons"].items():
            a, v = key.split("|")
            trs = []
            for r in c["rows"]:
                A, V = r["automap"], r["vlm"]
                trs.append(
                    f"<tr><td class='num'>{r['overlap']:.2f}</td>"
                    f"<td><b>{e(str(A.get('id') or ''))}</b><div>{_loc(A['box'], shape)}</div></td>"
                    f"<td>{e(', '.join(A['elements']))}</td><td>{e(A['material'])}<div class='sub'>match: {e(str(A['confidence']))}</div></td>"
                    f"<td><b>#{e(str(V.get('id') or ''))}</b><div>{_loc(V['box'], shape)}</div></td>"
                    f"<td>{e(', '.join(V['elements']))}</td><td>{e(V['material'])}<div class='sub'>self-rated: {e(str(V['confidence']))}</div>"
                    + (f"<div class='sub'>Alternatives: {e(_alts_text(V.get('alternatives')))}</div>" if V.get('alternatives') else "")
                    + "</td>"
                    f"<td class='sub'>{e(str(V.get('reasoning', '')))}</td>"
                    f"<td class='num'>{r['element_jaccard']:.2f}</td></tr>")
            body = ("<div class='scroll'><table class='phase'><tr><th>Overlap</th>"
                    f"<th>{e(_label(a))} region / location (px)</th>"
                    f"<th>{e(_label(a))} elements</th><th>{e(_label(a))} library match</th>"
                    f"<th>{e(_label(v))} region / location (px)</th>"
                    f"<th>{e(_label(v))} elements</th><th>LLM best guess</th><th>LLM reasoning</th>"
                    "<th>Element Jaccard</th></tr>" + "".join(trs) + "</table></div>") if trs else \
                   "<p class='note'>No regions matched.</p>"
            out.append(f"<details><summary>{e(_label(a))} vs {e(_label(v))} — {c['matched']} matched regions, "
                       f"side by side</summary>{body}</details>")
    return "".join(out)


def _alts_text(alts) -> str:
    """LLM alternative materials → 'name (note); name (note)'."""
    out = []
    for a in alts or []:
        if isinstance(a, dict):
            out.append(str(a.get("name", "")) + (f" ({a['note']})" if a.get("note") else ""))
        else:
            out.append(str(a))
    return "; ".join(x for x in out if x)


def _backfill_alternatives(results: dict, run_dir: pathlib.Path):
    """Runs made before 'alternatives' was kept in results.json: read them from each variant's
    <sample>_multi_llm_colocate.json on disk (matched by zone id). Read-only; the run folder is not modified."""
    for s, sd in results["samples"].items():
        ph = sd["phase"]
        for mid, regs in ph["regions"].items():
            if not mid.startswith("vlm:") or all("alternatives" in r for r in regs):
                continue
            f = run_dir / s / f"vlm_{mid[4:]}" / f"{s}_multi_llm_colocate.json"
            try:
                alts = {z.get("id"): z.get("alternatives", []) for z in json.loads(f.read_text())["overlap_zones"]}
            except Exception:
                continue
            for r in regs:
                r.setdefault("alternatives", alts.get(r.get("id"), []))
            for key, c in ph["comparisons"].items():
                if key.endswith("|" + mid):
                    for row in c["rows"]:
                        row["vlm"].setdefault("alternatives", alts.get(row["vlm"].get("id"), []))


def write_report(results: dict, out: pathlib.Path):
    _backfill_alternatives(results, out.parent)
    methods = results["methods"]
    samples = results["samples"]
    e = html.escape

    head = "<tr><th>Sample / element</th>" + "".join(f"<th>{e(_label(m))}</th>" for m in methods) + "</tr>"
    count_rows, time_rows = [], []
    for s, sd in samples.items():
        for el, d in sd["elements"].items():
            c_cells, t_cells = [], []
            for m in methods:
                flag = d.get("flags", {}).get(m)
                if m in d["errors"]:
                    c_cells.append(f'<td class="err" title="{e(d["errors"][m])}">error</td>')
                elif flag:
                    c_cells.append(f'<td class="err" title="{e(flag)}">⚠ {len(d["boxes"].get(m, []))}</td>')
                else:
                    c_cells.append(f"<td>{len(d['boxes'].get(m, []))}</td>")
                t = d.get("time_s", {}).get(m)
                t_cells.append(f"<td>{t:.1f}</td>" if t is not None else "<td>—</td>")
            count_rows.append(f"<tr><td>{e(s)} / {e(el)}</td>{''.join(c_cells)}</tr>")
            time_rows.append(f"<tr><td>{e(s)} / {e(el)}</td>{''.join(t_cells)}</tr>")

    opts = ['<option value="__all__">All elements (pooled)</option>'] + [
        f'<option value="{e(s)}/{e(el)}">{e(s)} / {e(el)}</option>'
        for s, sd in samples.items() for el in sd["elements"]]

    # viewer data: every element × method → its four image paths
    viewer = {"items": [], "images": {}, "counts": {}, "errors": {}, "boxes": {}, "shape": {}}
    for s_, sd in samples.items():
        for el, d in sd["elements"].items():
            key = f"{s_}/{el}"
            viewer["items"].append({"key": key, "label": f"{s_} / {el}  ({d['file']})"})
            viewer["images"][key] = d["images"]
            viewer["counts"][key] = {m: len(d["boxes"].get(m, [])) for m in methods}
            viewer["errors"][key] = d["errors"]
            viewer["boxes"][key] = d["boxes"]
            viewer["shape"][key] = d["shape"]
    materials = {s_: {"images": sd["phase"]["images"], "errors": sd["phase"]["errors"],
                      "shape": _sample_shape(sd),
                      "regions": {m: [{"id": e(str(r.get("id") or "")), "elements": r["elements"],
                                       "material": e(str(r["material"])), "confidence": e(str(r["confidence"])),
                                       "box": r["box"], "ratios": r.get("ratios") or {},
                                       "reasoning": e(str(r.get("reasoning") or "")),
                                       "alts": e(_alts_text(r.get("alternatives")))}
                                      for r in regs]
                                  for m, regs in sd["phase"]["regions"].items()}}
                 for s_, sd in samples.items()}

    prompts = "".join(f"<h3>{e(_label('vlm:' + v))}</h3><pre>{e(p)}</pre>"
                      for v, p in results.get("prompts", {}).items())
    ramp_legend = "".join(f'<span style="background:{c}"></span>' for c in _RAMP)
    model = results.get("vlm_model") or "none (--no-vlm)"
    mi = results.get("model_input", {})
    model_input_table = ('<details><summary>What each method was given (model input)</summary>'
                         '<div class="scroll"><table class="phase" style="margin-top:.5rem"><tr><th>Method</th>'
                         '<th>Model input</th></tr>'
                         + "".join(f"<tr><td>{e(_label(m))}</td><td>{e(mi.get(m, '?'))}</td></tr>" for m in methods)
                         + '</table></div></details>')
    data_json = json.dumps({"methods": methods, "vlm": results.get("vlm_variants", []), "model_input": mi,
                            "match_iou": results["match_iou"],
                            "phase_match_overlap": results.get("phase_match_overlap", 0.5),
                            "viewer": viewer, "materials": materials,
                            "overall_agreement": results["overall_agreement"],
                            "samples": {s: {"elements": {el: {"agreement": d["agreement"]}
                                                         for el, d in sd["elements"].items()}}
                                        for s, sd in samples.items()}})

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detection comparison — {e(results['run'])}</title><style>{_CSS}</style></head>
<body>
<header><div class="agency">HXN AutoMap · Benchmark</div>
<h1>AutoMap (OpenCV, StarDist) vs VLM / LLM</h1>
<div class="meta">Run {e(results['run'])} · {e(results['created'])} · VLM/LLM model: {e(model)} ·
AutoMap config: {e(pathlib.Path(results.get('automap_config', '')).name)}</div></header>
<nav><a href="#counts">Detections</a><a href="#agreement">Box agreement</a><a href="#images">Detection viewer</a>
<a href="#phase">Phase / material</a><a href="#reconcile">Material agreement (LLM)</a><a href="#calls">API calls &amp; cost</a><a href="#prompts">Prompts</a><a href="#timing">Timing</a></nav>
<main>
<section id="counts"><h2>Level 1 · Detections per element</h2>
<p class="note">AutoMap counts come from <code>MASTER_raw_detections.json</code>; VLM counts from each variant's
<code>_multi_features.json</code>. Hover an <i>error</i> cell for the message.
<span class="err">⚠</span> = the VLM reply was cut off at the token limit or could not be parsed, so the count is
incomplete — see <a href="#calls">API calls</a>.</p>
<div class="scroll"><table>{head}{''.join(count_rows)}</table></div></section>

<section id="agreement"><h2>Level 1 · Box agreement between methods</h2>
<p class="note">No hand-labelled ground truth yet, so methods are scored against each other.
Each cell is the box F1 between two methods: boxes are matched one-to-one when IoU ≥ {results['match_iou']},
F1 = 2 × matched / (boxes A + boxes B). 1.00 = identical detections, 0.00 = no overlap.</p>
<div class="controls"><label for="heat-sel">Show</label><select id="heat-sel">{''.join(opts)}</select></div>
<div class="scroll" id="heat"></div>
<div class="legend">0.0 <div class="ramp">{ramp_legend}</div> 1.0 &nbsp;F1 agreement</div></section>

<section id="images"><h2>Level 1 · Detection viewer</h2>
<p class="note"><b>Original</b> is the element TIFF min-max scaled to 0–255 — the least processing any method can use
(the TIFFs are float32 with values far below 1, so 8-bit images need this scaling). <b>Model input</b> is the exact image
each method analysed; the caption under each panel names every step applied. Images are enlarged with nearest-neighbour
for display only. With boxes on, <b>hover a box</b> to see its pixel
location on the TIFF (x/y range, center, size) and its best-overlapping box in the other panel, which is outlined in magenta.</p>
{model_input_table}
<div class="vbar">
  <button class="navbtn" id="v-prev" aria-label="Previous element">◀</button>
  <select id="v-el" aria-label="Element"></select>
  <button class="navbtn" id="v-next" aria-label="Next element">▶</button>
  <span class="seg" data-ctl="view" role="group" aria-label="Image"><button data-v="original">Original</button><button data-v="input">Model input</button></span>
  <span class="seg" data-ctl="boxes" role="group" aria-label="Boxes"><button data-v="on">Boxes on</button><button data-v="off">Boxes off</button></span>
  <span class="seg" data-ctl="tips" role="group" aria-label="Tooltips"><button data-v="on">Tooltips on</button><button data-v="off">Tooltips off</button></span>
  <span class="kbd">keys: ← → element · b boxes · i image · t tooltips</span>
</div>
<div class="panes">
  <div class="pane"><div class="phead"><label for="v-sel-a">Left</label><select id="v-sel-a"></select></div>
    <div class="imgwrap"><img id="v-img-a" alt=""><svg id="v-svg-a" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
    <div class="cap" id="v-cap-a"></div></div>
  <div class="pane"><div class="phead"><label for="v-sel-b">Right</label><select id="v-sel-b"></select></div>
    <div class="imgwrap"><img id="v-img-b" alt=""><svg id="v-svg-b" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
    <div class="cap" id="v-cap-b"></div></div>
</div>
<h3>All methods at a glance</h3>
<p class="note">Every method on the selected element, four per row. Hover a box: the best-overlapping box in every other
method is outlined in magenta (solid = matched, IoU ≥ {results['match_iou']}; dashed = overlaps but below that) and the tooltip lists them.
<b>Drag a tile</b> onto another to move it; click a tile to load it in the left panel above.</p>
<div class="vbar"><span class="seg" data-ctl="view" role="group" aria-label="Image"><button data-v="original">Original</button><button data-v="input">Model input</button></span> <span class="seg" data-ctl="boxes" role="group" aria-label="Boxes"><button data-v="on">Boxes on</button><button data-v="off">Boxes off</button></span> <span class="seg" data-ctl="tips" role="group" aria-label="Tooltips"><button data-v="on">Tooltips on</button><button data-v="off">Tooltips off</button></span> <button class="navbtn" id="v-grid-reset">Reset order</button></div>
<div class="mgrid" id="v-grid"></div></section>

<section id="phase"><h2>Level 2 · Phase / material regions</h2>
<p class="note">AutoMap: multi-element unions from <code>phase_detections.json</code> with their materials-library match.
VLM: LLM co-location zones from <code>_multi_llm_colocate.json</code> with the LLM's suggestion.
Regions are matched one-to-one when intersection ÷ smaller box ≥ {results.get('phase_match_overlap', 0.5)}
(AutoMap boxes cover the <i>union</i> of blobs, LLM boxes the <i>intersection</i> of elements, so plain IoU would under-match).</p>
<p class="caveat"><b>Material names are hypotheses, not identifications.</b> LLM materials are the model's <i>best guess</i> from which elements sit together (plus the sample context); its "high / medium / low" is the LLM's own self-rating, not a measured probability. AutoMap names are the closest entry in <code>materials_library.json</code> by element ratios. XRF gives elements only — no oxidation state, light elements (C, O, H) or crystal structure — so confirm phases with another technique (e.g. XANES, XRD, Raman); the LLM's <i>alternatives</i> say what signal would decide.</p>
<h3>Materials viewer</h3>
<p class="note">Pick two methods to compare their material regions on the all-element composite. The table under each image lists that method's regions with their pixel location. With boxes on, <b>hover a box</b>
for its location, elements, material, ratios / LLM reasoning, and its best-overlapping region in the other panel.</p>
<div class="vbar"><label for="m-sample">Sample</label><select id="m-sample"></select>
<span class="seg" data-ctl="boxes" role="group" aria-label="Boxes"><button data-v="on">Boxes on</button><button data-v="off">Boxes off</button></span> <span class="seg" data-ctl="tips" role="group" aria-label="Tooltips"><button data-v="on">Tooltips on</button><button data-v="off">Tooltips off</button></span></div>
<div class="panes">
  <div class="pane"><div class="phead"><label for="m-sel-a">Left</label><select id="m-sel-a"></select></div>
    <div class="imgwrap"><img id="m-img-a" alt=""><svg id="m-svg-a" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
    <div class="cap" id="m-cap-a"></div><div class="regions scroll" id="m-reg-a"></div></div>
  <div class="pane"><div class="phead"><label for="m-sel-b">Right</label><select id="m-sel-b"></select></div>
    <div class="imgwrap"><img id="m-img-b" alt=""><svg id="m-svg-b" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
    <div class="cap" id="m-cap-b"></div><div class="regions scroll" id="m-reg-b"></div></div>
</div>
<h3>All methods at a glance</h3>
<p class="note">Every method's material regions on the same composite, four per row. With boxes on, hover a box: the
best-overlapping region in every other method is outlined in magenta (solid = counts as matched, overlap ≥
{results.get('phase_match_overlap', 0.5)}; dashed = overlaps but below that), and the tooltip lists what each method
found at that spot. <b>Drag a tile</b> onto another to move it next to one you want to compare.
There is no model-input view here: these are all-element composites, not a per-method input image.</p>
<div class="vbar"><span class="seg" data-ctl="boxes" role="group" aria-label="Boxes"><button data-v="on">Boxes on</button><button data-v="off">Boxes off</button></span> <span class="seg" data-ctl="tips" role="group" aria-label="Tooltips"><button data-v="on">Tooltips on</button><button data-v="off">Tooltips off</button></span> <button class="navbtn" id="m-grid-reset">Reset order</button></div>
<div class="mgrid" id="m-grid"></div>
<h3>AutoMap vs VLM summary</h3>
{_phase_section(results)}</section>

<section id="reconcile"><h2>Level 3 · Material agreement (LLM review)</h2>
<p class="note">One LLM pass reads every method's material regions (location, elements, name) and groups the ones that
describe the same particle or phase — even when methods use different names or did not name it at all. Which methods
found or missed each group is then counted from those groups. Use it to see where deterministic and VLM methods agree,
disagree, or find things the other missed. The group names are also LLM best guesses — agreement between methods
means they found the same spot and elements, not that the material is confirmed.</p>
{_reconcile_section(results)}</section>



<section id="prompts"><h2>Prompts</h2><p class="note">System prompt sent with each VLM variant.</p>
{prompts or '<p class="note">VLM was not run.</p>'}</section>

<section id="calls"><h2>API calls &amp; cost</h2>
{_calls_section(results)}</section>

<section id="timing"><h2>Timing (seconds)</h2>
<p class="note">VLM wall time per call (calls run in parallel, so they overlap). AutoMap runs once per sample —
see the Phase section for its total time.</p>
<div class="scroll"><table>{head}{''.join(time_rows)}</table></div></section>
</main>
<div id="tip" role="tooltip"></div>
<script type="application/json" id="data">{data_json.replace('</', '<\\/')}</script>
<script>{_JS.replace('%RAMP%', json.dumps(_RAMP))}</script>
</body></html>"""
    out.write_text(page, encoding="utf-8")


# ── API calls & cost ─────────────────────────────────────────────────────────

def _reply_link(c: dict) -> str:
    if c.get("raw_reply"):
        return f'<a href="{html.escape(c["raw_reply"])}" target="_blank">reply</a>'
    return html.escape(c.get("error") or "")


def _calls_section(results: dict) -> str:
    e = html.escape
    calls = results.get("calls") or []
    if not calls:
        return '<p class="note">No API calls in this run (--no-vlm).</p>'
    order = results["methods"] + ["reconcile"]
    agg = {}
    for c in calls:
        a = agg.setdefault(c["method"], {"n": 0, "tin": 0, "tout": 0, "cost": 0.0, "cost_known": True,
                                         "cut": 0, "fail": 0, "err": 0})
        a["n"] += 1
        a["tin"] += c["input_tokens"] or 0
        a["tout"] += c["output_tokens"] or 0
        if c["cost_usd"] is None:
            a["cost_known"] = False
        else:
            a["cost"] += c["cost_usd"]
        a["cut"] += c["truncated"]
        a["fail"] += c["parse_status"] == "failed"
        a["err"] += bool(c["error"])
    rows, tot = [], {"n": 0, "tin": 0, "tout": 0, "cost": 0.0, "cut": 0, "fail": 0, "err": 0}
    for m in sorted(agg, key=lambda x: order.index(x) if x in order else 99):
        a = agg[m]
        for k in tot:
            tot[k] += a[k]
        bad = lambda v: f'<td class="err">{v}</td>' if v else "<td>0</td>"
        rows.append(f"<tr><td>{e(_label(m) if m != 'reconcile' else 'LLM material reconciliation')}</td>"
                    f"<td>{a['n']}</td><td>{a['tin']:,}</td><td>{a['tout']:,}</td>"
                    f"<td>{'$%.3f' % a['cost'] if a['cost_known'] else '—'}</td>"
                    f"{bad(a['cut'])}{bad(a['fail'])}{bad(a['err'])}</tr>")
    rows.append(f"<tr><th>Total</th><th>{tot['n']}</th><th>{tot['tin']:,}</th><th>{tot['tout']:,}</th>"
                f"<th>${tot['cost']:.3f}</th><th>{tot['cut']}</th><th>{tot['fail']}</th><th>{tot['err']}</th></tr>")
    detail = "".join(
        f"<tr><td>{e(c['kind'])}</td><td>{e(_label(c['method']) if c['method'] != 'reconcile' else 'reconcile')}</td>"
        f"<td>{e(c['element'] or '—')}</td><td>{e(str(c['stop_reason']))}</td>"
        f"<td{' class=err' if c['parse_status'] in ('failed', 'error') else ''}>{e(str(c['parse_status']))}</td>"
        f"<td>{c['input_tokens'] or '—'}</td><td>{c['output_tokens'] or '—'}</td>"
        f"<td>{'$%.4f' % c['cost_usd'] if c['cost_usd'] is not None else '—'}</td><td>{c['time_s']}</td>"
        f"<td>{_reply_link(c)}</td></tr>"
        for c in calls)
    price = results.get("pricing_per_mtok")
    return (f'<p class="note">Model: {e(str(results.get("vlm_model")))}'
            + (f' · priced at ${price[0]}/${price[1]} per 1M input/output tokens (Anthropic list price; '
               'a proxy may bill differently)' if price else ' · no price configured for this model')
            + '. <b>Cut off</b> = reply hit the token limit (results incomplete). '
              '<b>Parse failed</b> = reply was not usable JSON.</p>'
            '<div class="scroll"><table><tr><th>Method</th><th>Calls</th><th>Input tokens</th><th>Output tokens</th>'
            '<th>Cost</th><th>Cut off</th><th>Parse failed</th><th>Errors</th></tr>' + "".join(rows) + "</table></div>"
            '<details><summary>Every call</summary><div class="scroll"><table class="phase" style="margin-top:.5rem">'
            '<tr><th>Kind</th><th>Method</th><th>Element</th><th>Stop reason</th><th>Parse</th><th>In</th>'
            '<th>Out</th><th>Cost</th><th>s</th><th>Raw reply</th></tr>' + detail + "</table></div></details>")


# ── Level 3: LLM material reconciliation ─────────────────────────────────────

_AGREE_LABEL = {"all": "All methods", "det+vlm": "Deterministic + VLM", "det-only": "Deterministic only",
                "vlm-only": "VLM only", "single": "One method only"}


def _reconcile_section(results: dict) -> str:
    e = html.escape
    out = []
    for s, sd in results["samples"].items():
        rc = sd.get("reconcile")
        out.append(f"<h3>{e(s)}</h3>")
        if not rc:
            out.append('<p class="note">Not run (needs the VLM methods; disabled with --no-vlm).</p>')
            continue
        if rc.get("error"):
            out.append(f'<p class="note err">Reconciliation failed: {e(rc["error"])}</p>')
            continue
        methods = list(rc["per_method"])
        out.append(f'<p class="note"><b>LLM summary ({e(rc.get("model", ""))}):</b> {e(rc["overall"])}</p>')
        if rc["n_unassigned"]:
            out.append(f'<p class="note err">{rc["n_unassigned"]} region(s) were not assigned by the LLM and are '
                       'shown as their own groups.</p>')
        # class counts
        counts = {k: sum(g["agreement"] == k for g in rc["groups"]) for k in _AGREE_LABEL}
        out.append('<div class="scroll"><table><tr>' + "".join(f"<th>{e(v)}</th>" for v in _AGREE_LABEL.values())
                   + "<th>Total groups</th></tr><tr>" + "".join(f"<td>{counts[k]}</td>" for k in _AGREE_LABEL)
                   + f"<td>{len(rc['groups'])}</td></tr></table></div>")
        # per-method table
        rows = "".join(
            f"<tr><td>{e(_label(m))}</td><td>{d['regions']}</td><td>{d['groups']}</td><td>{d['shared']}</td>"
            f"<td>{d['unique']}</td><td>{d['shared_with_det']}</td><td>{d['shared_with_vlm']}</td></tr>"
            for m, d in rc["per_method"].items())
        out.append('<h3>Per method</h3><p class="note">Groups = distinct materials this method found. '
                   'Shared = also found by at least one other method; unique = found by nobody else.</p>'
                   '<div class="scroll"><table><tr><th>Method</th><th>Regions</th><th>Groups</th><th>Shared</th>'
                   '<th>Unique</th><th>Shared with a deterministic method</th><th>Shared with a VLM</th></tr>'
                   + rows + "</table></div>")
        # pairwise heatmap (Jaccard of group sets)
        head = "<tr><th></th>" + "".join(f'<th class="col">{e(_label(m))}</th>' for m in methods) + "</tr>"
        hr = []
        for i, a in enumerate(methods):
            cells = []
            for j, b in enumerate(methods):
                if i == j:
                    cells.append('<td class="diag">—</td>')
                    continue
                pw = rc["pairwise"][f"{a}|{b}" if i < j else f"{b}|{a}"]
                bg, ink = _ramp(pw["jaccard"])
                cells.append(f'<td class="cell" style="background:{bg};color:{ink}" '
                             f'title="{e(_label(a))} & {e(_label(b))}: {pw["both"]} shared materials, '
                             f'Jaccard {pw["jaccard"]:.2f}">{pw["jaccard"]:.2f}</td>')
            hr.append(f"<tr><th>{e(_label(a))}</th>{''.join(cells)}</tr>")
        out.append('<h3>Pairwise material agreement</h3><p class="note">Share of materials two methods have in common '
                   '(Jaccard: shared ÷ found by either). Hover for the count.</p>'
                   f'<div class="scroll"><table class="heat">{head}{"".join(hr)}</table></div>')
        # groups table with filter
        opts = '<option value="">All groups</option>' + "".join(
            f'<option value="{k}">{e(v)} ({counts[k]})</option>' for k, v in _AGREE_LABEL.items())
        grows = []
        for g in rc["groups"]:
            found = "".join(f'<li><b>{e(_label(x["method"]))}</b> {e(x["region_id"])}: '
                            f'{e(x["name_used"] or "unnamed")} <span class="sub">({e(",".join(x["elements"]))})</span></li>'
                            for x in g["members"])
            missing = ", ".join(e(_label(m)) for m in g["missing_from"]) or "—"
            grows.append(f'<tr data-a="{g["agreement"]}"><td class="num">{g["id"]}</td>'
                         f'<td><b>{e(g["material"])}</b><div class="sub">{e(", ".join(g["elements"]))}</div></td>'
                         f'<td>{e(_AGREE_LABEL[g["agreement"]])}</td><td><ul class="mlist">{found}</ul></td>'
                         f'<td class="sub">{missing}</td><td class="sub">{e(g["note"])}</td></tr>')
        out.append(f'<h3>Material groups</h3><div class="controls"><label>Show</label>'
                   f'<select class="grp-filter" data-target="grp-{e(s)}">{opts}</select></div>'
                   f'<div class="scroll"><table class="phase" id="grp-{e(s)}"><tr><th>#</th><th>Material (LLM best guess)</th>'
                   '<th>Agreement</th><th>Found by (region: name that method used)</th><th>Missing from</th><th>Note</th></tr>'
                   + "".join(grows) + "</table></div>")
        out.append(f'<p class="note">Files: <code>{e(s)}/material_reconciliation.json</code>, prompt and raw reply '
                   f'alongside it.</p>')
    return "".join(out)


# ── runs index ───────────────────────────────────────────────────────────────

def write_runs_index(runs_dir: pathlib.Path):
    """benchmarks/runs/index.html — one row per run, newest first."""
    e = html.escape
    rows = []
    for rd in sorted((d for d in runs_dir.iterdir() if (d / "results.json").exists()), reverse=True):
        try:
            r = json.loads((rd / "results.json").read_text())
        except Exception:
            continue
        calls = r.get("calls") or []
        cost = sum(c["cost_usd"] or 0 for c in calls)
        cut = sum(c["truncated"] for c in calls)
        bad = sum(c["parse_status"] in ("failed", "error") for c in calls)
        boxes = {m: 0 for m in r["methods"]}
        mats = {m: 0 for m in r["methods"]}
        groups, shared = 0, 0
        for sd in r["samples"].values():
            for d in sd["elements"].values():
                for m in boxes:
                    boxes[m] += len(d["boxes"].get(m, []))
            for m in mats:
                mats[m] += len(sd["phase"]["regions"].get(m, []))
            rc = sd.get("reconcile") or {}
            groups += len(rc.get("groups", []))
            shared += sum(g["agreement"] in ("all", "det+vlm") for g in rc.get("groups", []))
        per = "".join(f"<li>{e(_label(m))}: {boxes[m]} boxes · {mats[m]} materials</li>" for m in r["methods"])
        rows.append(
            f'<tr><td><a href="{e(rd.name)}/index.html">{e(rd.name)}</a></td><td>{e(r.get("created", ""))}</td>'
            f'<td>{e(str(r.get("vlm_model") or "—"))}</td><td>{e(", ".join(r["samples"]))}</td>'
            f'<td>{len(calls)}</td><td>${cost:.2f}</td>'
            f'<td{" class=err" if cut or bad else ""}>{cut} cut off · {bad} failed</td>'
            f'<td>{groups} groups · {shared} found by det + VLM</td>'
            f'<td><details><summary>{len(r["methods"])} methods</summary><ul class="mlist">{per}</ul></details></td></tr>')
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Benchmark runs</title>
<style>{_CSS}</style></head><body>
<header><div class="agency">HXN AutoMap · Benchmark</div><h1>All comparison runs</h1>
<div class="meta">Newest first · regenerated after every run</div></header>
<main><section><div class="scroll"><table class="phase"><tr><th>Run</th><th>Created</th><th>Model</th><th>Samples</th>
<th>API calls</th><th>Cost</th><th>Reply problems</th><th>Material agreement</th><th>Per method</th></tr>
{''.join(rows) or '<tr><td colspan=9>No runs yet.</td></tr>'}</table></div></section></main></body></html>"""
    (runs_dir / "index.html").write_text(page, encoding="utf-8")
