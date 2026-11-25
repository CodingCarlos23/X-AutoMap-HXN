#!/usr/bin/env python3
"""
Car Standalone cell-size sweep — VLM raw vs car mosaic at 1, 2, 3, 4, 5, 10 px (only the cell size changes).
Built only from a run's results.json (no API calls, rerun freely).

  Section 1  Detection: boxes per size (chart + per-element table), agreement of each size with VLM raw,
             per-element image grid (one panel per method)
  Section 2  Materials: LLM material regions per size, location matches with raw, composite grid

Usage:
  pixi run python benchmarks/car_cell_sweep_report.py benchmarks/runs/<timestamp>_car_cell_sweep
Writes <run>/car_cell_sweep.html.
"""
import argparse
import html
import json
import pathlib
import re

from report import GRP_CSS, TIP_CSS, grouped_table
from car_vs_opencv_report import el_list, svg_boxes, t_head, t_loc, t_row, t_warn, unnamed
from level3 import _greedy

RAW = "vlm:raw"
e = html.escape


def cell(m):
    x = re.match(r"vlm:car_standalone_(\d+)px$", m)
    return int(x.group(1)) if x else None


def name(m):
    return "VLM raw" if m == RAW else f"Car {cell(m)} px"


def line_chart(points, ref, ymax):
    """Total boxes vs cell size (car variants) with VLM raw as a dashed reference line."""
    W, H, l, r, t, b = 760, 260, 50, 20, 16, 40
    xs = [p[0] for p in points]
    xmin, xmax = min(xs), max(xs)
    X = lambda v: l + (v - xmin) / ((xmax - xmin) or 1) * (W - l - r)
    Y = lambda v: H - b - v / (ymax or 1) * (H - t - b)
    out = [f'<svg class=chart viewBox="0 0 {W} {H}" role=img>']
    for k in range(5):
        v = ymax * k / 4
        out.append(f'<line class=grid x1="{l}" x2="{W - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                   f'<text class=ax x="{l - 6}" y="{Y(v) + 4:.1f}" text-anchor=end>{v:.0f}</text>')
    for v, _ in points:
        out.append(f'<text class=ax x="{X(v):.1f}" y="{H - b + 16}" text-anchor=middle>{v} px</text>')
    out.append(f'<text class=ax x="{(l + W - r) / 2:.0f}" y="{H - 6}" text-anchor=middle>car cell size</text>')
    if ref is not None:
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" stroke="var(--s-raw)" '
                   f'stroke-width="2" stroke-dasharray="6 4"/><text class="val" x="{l + 6}" y="{Y(ref) + 15:.1f}" '
                   f'text-anchor="start">VLM raw: {ref}</text>')
    pts = " ".join(f"{X(v):.1f},{Y(n):.1f}" for v, n in points)
    out.append(f'<polyline points="{pts}" fill="none" stroke="var(--s-car)" stroke-width="2"/>')
    for v, n in points:
        out.append(f'<circle cx="{X(v):.1f}" cy="{Y(n):.1f}" r="5" fill="var(--s-car)" stroke="var(--surface-1)" '
                   f'stroke-width="2" data-tip="{e(t_head(f"Car {v} px") + t_row("Boxes (all elements):", str(n)))}"/>'
                   f'<text class="val" x="{X(v):.1f}" y="{Y(n) - 10:.1f}" text-anchor="middle">{n}</text>')
    out.append("</svg>")
    return "".join(out)


def build(run_dir: pathlib.Path) -> pathlib.Path:
    r = json.loads((run_dir / "results.json").read_text())
    s_name = next(iter(r["samples"]))
    smp = r["samples"][s_name]
    els = smp["elements"]
    methods = [m for m in r.get("vlm_variants", []) if m == RAW or cell(m)]
    cars = sorted([m for m in methods if cell(m)], key=cell)
    order = ([RAW] if RAW in methods else []) + cars
    thr, pthr = r.get("match_iou", 0.25), r.get("phase_match_overlap", 0.5)
    calls = r.get("calls", [])
    cost = sum(c.get("cost_usd") or 0 for c in calls)
    bad = sum(1 for c in calls if c.get("error") or c.get("truncated") or c.get("parse_status") == "failed")
    mi = r.get("model_input", {})

    # ── Section 1 ──
    tot = {m: sum(len(d["boxes"].get(m, [])) for d in els.values()) for m in order}
    count_rows = [(f"<b>{e(en)}</b>", [[len(d["boxes"].get(m, [])) for m in order]]) for en, d in els.items()]
    t_counts = grouped_table("Element", [("Boxes found", "reg", [e(name(m)) for m in order])], count_rows,
                             ("All elements", [[tot[m] for m in order]]))
    agree_rows = []
    for m in cars:
        mt = n_iw = 0
        for d in els.values():
            a = d.get("agreement", {}).get(f"{RAW}|{m}") or d.get("agreement", {}).get(f"{m}|{RAW}") or {}
            mt += a.get("matched", 0); n_iw += a.get("matched", 0) * a.get("mean_iou", 0.0)
        n = tot.get(RAW, 0) + tot[m]
        agree_rows.append((e(name(m)), [[tot[m]], [mt, f"{2 * mt / n if n else 0:.2f}", f"{n_iw / mt if mt else 0:.2f}"]]))
    t_agree = grouped_table("Method", [("", "info", ["Boxes"]),
                                       (f"Agreement with VLM raw ({tot.get(RAW, 0)} boxes · IoU ≥ {thr})", "loc",
                                        ["Matched", "F1", "Mean IoU"])], agree_rows) if RAW in tot else ""
    ymax = max(list(tot.values()) + [1]) * 1.12
    chart = line_chart([(cell(m), tot[m]) for m in cars], tot.get(RAW), ymax)

    details = []
    for en, d in els.items():
        base = d["images"].get(RAW, {}).get("original") or next((v.get("original") for v in d["images"].values()), "")
        shape = d["shape"]
        panes = []
        for m in order:
            bx = d["boxes"].get(m, [])
            ov = svg_boxes({("raw" if m == RAW else "car"): [
                (b, t_head(f"{e(name(m))} · box #{k + 1} of {len(bx)}") + t_loc(b, shape) + t_warn()
                 + t_row("Element:", el_list([en]))) for k, b in enumerate(bx)]})
            inp = d["images"].get(m, {}).get("input", "")
            err = d.get("errors", {}).get(m) or d.get("flags", {}).get(m)
            panes.append(f'<div class=pane><div class=ph><b>{e(name(m))}</b> · {len(bx)}'
                         + (f' · <a href="{e(inp)}" target=_blank>input ↗</a>' if inp else "")
                         + (f' <span class=bad title="{e(err)}">⚠</span>' if err else "")
                         + f'</div><div class=pic><img loading=lazy src="{e(base)}" alt="">{ov}</div></div>')
        details.append(f'<details><summary>{e(en)} <span class=mut>— '
                       + " · ".join(f"{e(name(m))} {len(d['boxes'].get(m, []))}" for m in order)
                       + f'</span></summary><div class=grid7>{"".join(panes)}</div></details>')

    # ── Section 2 ──
    regs = {m: smp["phase"]["regions"].get(m, []) for m in order}
    shape0 = next(iter(els.values()))["shape"]
    mat_rows = []
    for m in order:
        named = sum(1 for x in regs[m] if not unnamed(x.get("material")))
        loc = len(_greedy(regs[RAW], regs[m], pthr)) if (RAW in regs and m != RAW) else "—"
        mat_rows.append((e(name(m)), [[len(regs[m]), named], [loc]]))
    t_mat = grouped_table("Method", [("Material regions", "reg", ["Regions", "Named"]),
                                     (f"Same location as VLM raw (≥ {pthr:.0%} overlap)", "loc", ["Matched regions"])],
                          mat_rows)
    mpanes = []
    for m in order:
        comp = smp["phase"].get("images", {}).get(m, {}).get("plain", "")
        tips = []
        for x in regs[m]:
            alts = "; ".join(a.get("name", "") if isinstance(a, dict) else str(a) for a in x.get("alternatives") or [])
            tips.append((x["box"], t_head(f'{e(name(m))} · region {e(str(x.get("id", "")))}') + t_loc(x["box"], shape0)
                         + t_warn() + t_row("Elements:", el_list(x.get("elements")))
                         + t_row("LLM Best Guess:", f'{e(x.get("material") or "")} <span class="t-dim">(self-rated '
                                                    f'{e(str(x.get("confidence", "")))})</span>')
                         + (t_row("Reasoning:", e(x["reasoning"])) if x.get("reasoning") else "")
                         + (t_row("Alternatives:", e(alts)) if alts else "")))
        ov = svg_boxes({("raw" if m == RAW else "car"): tips})
        mpanes.append(f'<div class=pane><div class=ph><b>{e(name(m))}</b> · {len(regs[m])} regions</div>'
                      f'<div class=pic><img loading=lazy src="{e(comp)}" alt="">{ov}</div></div>')
    rlists = "".join(
        f'<details><summary>{e(name(m))} — {len(regs[m])} material regions</summary><ol class=rl>' + "".join(
            f'<li><b>{"<i>unnamed</i>" if unnamed(x.get("material")) else e(x.get("material") or "")}</b> '
            f'<span class=mut>({e(str(x.get("confidence", "")))}) · {e(", ".join(x.get("elements") or []))}</span>'
            + (f'<div class=mut>{e(x["reasoning"])}</div>' if x.get("reasoning") else "") + "</li>"
            for x in regs[m]) + "</ol></details>" for m in order)
    model_list = "".join(f"<li><b>{e(name(m))}</b>: {e(mi.get(m, ''))}</li>" for m in order)

    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>Car cell-size sweep — {e(r.get("run", run_dir.name))}</title>
<style>
.viz-root{{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#dcdbd6;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--text-muted:#77766f;--grid:#e6e5e0;--s-raw:#6250d6;--s-car:#eb6834;--warn:#8a5a00;--warn-bg:#fff4dc}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9b9a90;--grid:#2c2b28;--s-raw:#9085e9;--s-car:#d95926;--warn:#f0c060;--warn-bg:#3a2e12}}}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,sans-serif}}
.viz-root{{background:var(--surface-0);color:var(--text-primary);min-height:100vh;padding:24px 32px 64px}}
h1{{margin:0 0 4px;font-size:22px}} h2{{margin:30px 0 6px;font-size:18px}} h4{{margin:0 0 6px}}
.mut{{color:var(--text-muted);font-weight:normal}} p.lead{{color:var(--text-secondary);max-width:1000px}}
.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}
.tile{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:8px 14px;min-width:130px}}
.tile b{{display:block;font-size:20px}} .tile span{{color:var(--text-secondary);font-size:12px}}
.panel{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:10px 0;max-width:820px}}
.tw{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;overflow-x:auto;margin:10px 0;width:fit-content;max-width:100%}}
.tw table.grp{{font-size:13px}}
.chart{{width:100%;height:auto;display:block}} .chart .grid{{stroke:var(--grid)}}
.chart .ax,.chart .val{{fill:var(--text-secondary);font-size:11px}}
details{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;margin:8px 0;padding:8px 12px}}
details>summary{{cursor:pointer;font-weight:600}}
.grid7{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;margin-top:10px}}
.pane{{background:var(--surface-0);border:1px solid var(--border);border-radius:6px;padding:6px}}
.ph{{font-size:12.5px;margin-bottom:4px}} .ph a{{color:var(--text-secondary)}} .bad{{color:#c0392b}}
.pic{{position:relative;line-height:0}} .pic img{{width:100%;image-rendering:pixelated}}
.ov{{position:absolute;inset:0;width:100%;height:100%}}
.ov rect{{fill:transparent;stroke-width:1.6;vector-effect:non-scaling-stroke}} .ov rect:hover{{stroke-width:3.5}}
.g-raw rect{{stroke:var(--s-raw)}} .g-car rect{{stroke:var(--s-car)}}
.rl{{font-size:13px;padding-left:22px}} .rl li{{margin-bottom:5px}}
.warn{{background:var(--warn-bg);color:var(--warn);border-radius:4px;padding:6px 10px;font-size:12.5px;margin:8px 0}}
ul.mi{{font-size:13px;color:var(--text-secondary);margin:6px 0}}
{TIP_CSS}
{GRP_CSS}
</style></head>
<body><div class=viz-root>
<h1>Car Standalone cell-size sweep <span class=mut>— {e(s_name)}, {len(els)} elements</span></h1>
<p class=lead>Run <code>{e(r.get("run", run_dir.name))}</code> · {e(r.get("created", ""))} · model <code>{e(r.get("vlm_model") or "")}</code>.
Only the car cell size changes (same stretch, model and prompt wording); VLM raw is the baseline. No OpenCV detection
in any method.</p>
<ul class=mi>{model_list}</ul>
<div class=tiles><div class=tile><b>{len(calls)}</b><span>API calls</span></div>
<div class=tile><b>${cost:.2f}</b><span>estimated cost</span></div>
<div class=tile><b>{bad}</b><span>failed / cut-off / unparsed replies</span></div></div>

<h2>1 · Detection</h2>
<div class=panel><h4>Total boxes vs car cell size <span class=mut>(all {len(els)} elements; dashed = VLM raw)</span></h4>{chart}</div>
<h4>Agreement of each size with VLM raw</h4>
<p class=lead>Agreement, not accuracy: a car box matches a raw box when IoU ≥ {thr}. F1 = 2 × matched ÷ (raw boxes + car boxes).</p>
<div class=tw>{t_agree}</div>
<h4>Boxes per element</h4>
<div class=tw>{t_counts}</div>
<h4>Side by side per element <span class=mut>— every panel is the same normalized scan with that method's boxes; "input" opens what the VLM saw</span></h4>
{"".join(details)}

<h2>2 · Materials</h2>
<p class=lead>Per method, one LLM call reads that method's boxes for all elements and names the multi-element regions
(same prompt for every method).</p>
<div class=warn>⚠ LLM results may be inaccurate — they need manual review and correction.</div>
<div class=tw>{t_mat}</div>
<div class=grid7>{"".join(mpanes)}</div>
{rlists}
<div id=tip></div></div>
<script>
const tip=document.getElementById('tip');
document.addEventListener('mousemove',ev=>{{const t=ev.target.closest('[data-tip]');
  if(!t){{tip.style.display='none';return}} tip.innerHTML=t.dataset.tip;tip.style.display='block';
  const r=tip.getBoundingClientRect();
  tip.style.left=Math.min(ev.clientX+14,innerWidth-r.width-8)+'px';tip.style.top=Math.min(ev.clientY+14,innerHeight-r.height-8)+'px';}});
</script></body></html>"""
    out = run_dir / "car_cell_sweep.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    print("Report →", build(ap.parse_args().run_dir.resolve()))
