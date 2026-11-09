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
.pane .cap { font-size: .85rem; color: var(--text-secondary); margin-top: .35rem; }
.pane .cap b { color: var(--text-primary); }
.pane .cap .err { color: var(--error); }
.strip { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .6rem; margin-top: 1rem; }
.strip figure { cursor: pointer; border: 2px solid transparent; border-radius: 6px; padding: 2px; }
.strip figure.on { border-color: var(--blue-60); }
.strip img { image-rendering: pixelated; }
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

/* ---------- detection viewer ---------- */
const V = DATA.viewer, M = DATA.methods;
const st = { idx: 0, boxes: true, view: 'original', a: M[0], b: DATA.vlm[0] || M[1] || M[0] };
const $ = id => document.getElementById(id);
function fillSelect(sel, val) { sel.innerHTML = M.map(m => `<option value="${m}"${m === val ? ' selected' : ''}>${label(m)}</option>`).join(''); }
function pick(imgs, view, boxes) {
  const mode = (view === 'input' && imgs.input) ? 'input' : 'original';
  return { src: imgs[mode + (boxes ? '_boxes' : '')], mode };
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
function renderViewer() {
  const item = V.items[st.idx];
  $('v-el').value = String(st.idx);
  for (const k of ['a', 'b']) {
    const [src, cap] = paneHTML(st[k], item);
    $('v-img-' + k).src = src || ''; $('v-img-' + k).alt = cap.replace(/<[^>]+>/g, '');
    $('v-cap-' + k).innerHTML = cap;
  }
  $('v-strip').innerHTML = M.map(m => {
    const [src, cap] = paneHTML(m, item);
    return `<figure data-m="${m}" class="${m === st.a ? 'on' : ''}"><img loading="lazy" src="${src || ''}" alt="${label(m)}">
            <figcaption><b>${label(m)}</b><span>${(V.counts[item.key] || {})[m] ?? ''}</span></figcaption></figure>`;
  }).join('');
  document.querySelectorAll('#v-boxes button').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.v === 'on') === st.boxes)));
  document.querySelectorAll('#v-view button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === st.view)));
}
$('v-el').innerHTML = V.items.map((it, i) => `<option value="${i}">${it.label}</option>`).join('');
fillSelect($('v-sel-a'), st.a); fillSelect($('v-sel-b'), st.b);
$('v-el').onchange = e => { st.idx = +e.target.value; renderViewer(); };
$('v-prev').onclick = () => { st.idx = (st.idx - 1 + V.items.length) % V.items.length; renderViewer(); };
$('v-next').onclick = () => { st.idx = (st.idx + 1) % V.items.length; renderViewer(); };
$('v-sel-a').onchange = e => { st.a = e.target.value; renderViewer(); };
$('v-sel-b').onchange = e => { st.b = e.target.value; renderViewer(); };
document.querySelectorAll('#v-boxes button').forEach(b => b.onclick = () => { st.boxes = b.dataset.v === 'on'; renderViewer(); renderMat(); });
document.querySelectorAll('#v-view button').forEach(b => b.onclick = () => { st.view = b.dataset.v; renderViewer(); });
$('v-strip').onclick = e => { const f = e.target.closest('figure'); if (f) { st.a = f.dataset.m; $('v-sel-a').value = st.a; renderViewer(); } };
document.addEventListener('keydown', e => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowRight') $('v-next').click();
  else if (e.key === 'ArrowLeft') $('v-prev').click();
  else if (e.key === 'b') { st.boxes = !st.boxes; renderViewer(); renderMat(); }
  else if (e.key === 'i') { st.view = st.view === 'input' ? 'original' : 'input'; renderViewer(); }
});
renderViewer();

/* ---------- material-group filter ---------- */
document.querySelectorAll('.grp-filter').forEach(sel => sel.onchange = () => {
  document.querySelectorAll('#' + CSS.escape(sel.dataset.target) + ' tr[data-a]').forEach(tr =>
    tr.style.display = (!sel.value || tr.dataset.a === sel.value) ? '' : 'none');
});

/* ---------- materials viewer ---------- */
const MAT = DATA.materials, MS = Object.keys(MAT);
const mt = { s: MS[0], a: M[0], b: DATA.vlm[0] || M[1] || M[0] };
function regTable(regs) {
  if (!regs || !regs.length) return '<p class="note">No multi-element regions.</p>';
  return '<table class="phase"><tr><th>#</th><th>Elements</th><th>Material</th><th>Confidence</th></tr>' +
    regs.map(r => `<tr><td>${r.id ?? ''}</td><td>${r.elements.join(', ')}</td><td>${r.material}</td><td>${r.confidence}</td></tr>`).join('') + '</table>';
}
function renderMat() {
  if (!MS.length) return;
  const d = MAT[mt.s];
  for (const k of ['a', 'b']) {
    const m = mt[k], imgs = d.images[m] || {}, regs = d.regions[m] || [], err = d.errors[m];
    $('m-img-' + k).src = (st.boxes ? imgs.boxes : imgs.plain) || '';
    $('m-cap-' + k).innerHTML = `<b>${label(m)}</b> · ${err ? '<span class="err">error</span>' : regs.length + ' material regions'} · ${st.boxes ? 'with boxes' : 'no boxes'}`;
    $('m-reg-' + k).innerHTML = regTable(regs);
  }
  document.querySelectorAll('#m-boxes button').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.v === 'on') === st.boxes)));
}
if (MS.length) {
  $('m-sample').innerHTML = MS.map(s => `<option>${s}</option>`).join('');
  $('m-sample').onchange = e => { mt.s = e.target.value; renderMat(); };
  fillSelect($('m-sel-a'), mt.a); fillSelect($('m-sel-b'), mt.b);
  $('m-sel-a').onchange = e => { mt.a = e.target.value; renderMat(); };
  $('m-sel-b').onchange = e => { mt.b = e.target.value; renderMat(); };
  document.querySelectorAll('#m-boxes button').forEach(b => b.onclick = () => { st.boxes = b.dataset.v === 'on'; renderMat(); renderViewer(); });
  renderMat();
}
"""




def _label(m: str) -> str:
    return (m.replace("opencv:", "OpenCV ").replace("vlm:", "VLM ").replace("stardist", "StarDist")
             .replace(":", " · "))


def _ramp(v: float) -> tuple:
    i = min(len(_RAMP) - 1, int(v * len(_RAMP)))
    return _RAMP[i], ("#ffffff" if i >= 7 else "#1b1b1b")


def _phase_section(results: dict) -> str:
    e = html.escape
    am, vv = results.get("automap_methods", []), results.get("vlm_variants", [])
    out = []
    for s, sd in results["samples"].items():
        ph = sd["phase"]
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
                    f"<td>{e(', '.join(A['elements']))}</td><td>{e(A['material'])}<div class='sub'>{e(str(A['confidence']))}</div></td>"
                    f"<td>{e(', '.join(V['elements']))}</td><td>{e(V['material'])}<div class='sub'>{e(str(V['confidence']))}</div></td>"
                    f"<td class='sub'>{e(str(V.get('reasoning', '')))}</td>"
                    f"<td class='num'>{r['element_jaccard']:.2f}</td></tr>")
            body = ("<div class='scroll'><table class='phase'><tr><th>Overlap</th>"
                    f"<th>{e(_label(a))} elements</th><th>{e(_label(a))} library match</th>"
                    f"<th>{e(_label(v))} elements</th><th>LLM suggestion</th><th>LLM reasoning</th>"
                    "<th>Element Jaccard</th></tr>" + "".join(trs) + "</table></div>") if trs else \
                   "<p class='note'>No regions matched.</p>"
            out.append(f"<details><summary>{e(_label(a))} vs {e(_label(v))} — {c['matched']} matched regions, "
                       f"side by side</summary>{body}</details>")
    return "".join(out)


def write_report(results: dict, out: pathlib.Path):
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
    viewer = {"items": [], "images": {}, "counts": {}, "errors": {}}
    for s_, sd in samples.items():
        for el, d in sd["elements"].items():
            key = f"{s_}/{el}"
            viewer["items"].append({"key": key, "label": f"{s_} / {el}  ({d['file']})"})
            viewer["images"][key] = d["images"]
            viewer["counts"][key] = {m: len(d["boxes"].get(m, [])) for m in methods}
            viewer["errors"][key] = d["errors"]
    materials = {s_: {"images": sd["phase"]["images"], "errors": sd["phase"]["errors"],
                      "regions": {m: [{"id": r.get("id"), "elements": r["elements"],
                                       "material": e(str(r["material"])), "confidence": e(str(r["confidence"]))}
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
for display only. Click a thumbnail to put it in the left panel.</p>
{model_input_table}
<div class="vbar">
  <button class="navbtn" id="v-prev" aria-label="Previous element">◀</button>
  <select id="v-el" aria-label="Element"></select>
  <button class="navbtn" id="v-next" aria-label="Next element">▶</button>
  <span class="seg" id="v-view" role="group" aria-label="Image"><button data-v="original">Original</button><button data-v="input">Model input</button></span>
  <span class="seg" id="v-boxes" role="group" aria-label="Boxes"><button data-v="on">Boxes on</button><button data-v="off">Boxes off</button></span>
  <span class="kbd">keys: ← → element · b boxes · i image</span>
</div>
<div class="panes">
  <div class="pane"><div class="phead"><label for="v-sel-a">Left</label><select id="v-sel-a"></select></div>
    <img id="v-img-a" alt=""><div class="cap" id="v-cap-a"></div></div>
  <div class="pane"><div class="phead"><label for="v-sel-b">Right</label><select id="v-sel-b"></select></div>
    <img id="v-img-b" alt=""><div class="cap" id="v-cap-b"></div></div>
</div>
<div class="strip" id="v-strip"></div></section>

<section id="phase"><h2>Level 2 · Phase / material regions</h2>
<p class="note">AutoMap: multi-element unions from <code>phase_detections.json</code> with their materials-library match.
VLM: LLM co-location zones from <code>_multi_llm_colocate.json</code> with the LLM's suggestion.
Regions are matched one-to-one when intersection ÷ smaller box ≥ {results.get('phase_match_overlap', 0.5)}
(AutoMap boxes cover the <i>union</i> of blobs, LLM boxes the <i>intersection</i> of elements, so plain IoU would under-match).</p>
<h3>Materials viewer</h3>
<p class="note">Pick two methods to compare their material regions on the all-element composite. The table under each image lists that method's regions.</p>
<div class="vbar"><label for="m-sample">Sample</label><select id="m-sample"></select>
<span class="seg" id="m-boxes" role="group" aria-label="Boxes"><button data-v="on">Boxes on</button><button data-v="off">Boxes off</button></span></div>
<div class="panes">
  <div class="pane"><div class="phead"><label for="m-sel-a">Left</label><select id="m-sel-a"></select></div>
    <img id="m-img-a" alt=""><div class="cap" id="m-cap-a"></div><div class="regions scroll" id="m-reg-a"></div></div>
  <div class="pane"><div class="phead"><label for="m-sel-b">Right</label><select id="m-sel-b"></select></div>
    <img id="m-img-b" alt=""><div class="cap" id="m-cap-b"></div><div class="regions scroll" id="m-reg-b"></div></div>
</div>
<h3>AutoMap vs VLM summary</h3>
{_phase_section(results)}</section>

<section id="reconcile"><h2>Level 3 · Material agreement (LLM review)</h2>
<p class="note">One LLM pass reads every method's material regions (location, elements, name) and groups the ones that
describe the same particle or phase — even when methods use different names or did not name it at all. Which methods
found or missed each group is then counted from those groups. Use it to see where deterministic and VLM methods agree,
disagree, or find things the other missed.</p>
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
                   f'<div class="scroll"><table class="phase" id="grp-{e(s)}"><tr><th>#</th><th>Material (LLM canonical)</th>'
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
