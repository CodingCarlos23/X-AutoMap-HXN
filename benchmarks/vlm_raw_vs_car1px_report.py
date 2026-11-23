#!/usr/bin/env python3
"""
VLM raw image vs Car Standalone 1 px — built only from a run's results.json (no API calls, rerun freely).

  Section 1  Detection:  vlm:raw boxes vs vlm:car_standalone_1px boxes, per element, side by side
  Section 2  Materials:  each method's LLM material regions (co-location step), side by side + location matches

Usage:
  pixi run python benchmarks/vlm_raw_vs_car1px_report.py benchmarks/runs/<timestamp>_raw_vs_car1px
Writes <run>/raw_vs_car1px.html (images are referenced relative to the run folder).
"""
import argparse
import html
import json
import pathlib

from report import GRP_CSS, TIP_CSS, grouped_table
from car_vs_opencv_report import _iou, bar_chart, el_list, svg_boxes, t_head, t_loc, t_row, t_warn, unnamed
from level3 import _greedy

RAW, CAR = "vlm:raw", "vlm:car_standalone_1px"
NAME = {RAW: "VLM raw image", CAR: "Car 1 px"}
CLS = {RAW: "raw", CAR: "car"}
e = html.escape


def det_tip(m, k, boxes, o, oboxes, el, shape, thr):
    b = boxes[k]
    best = max(((_iou(b, x), j) for j, x in enumerate(oboxes)), default=(0.0, None))
    ov = (f'box #{best[1] + 1}, IoU {best[0]:.2f} — '
          + ('<span class="t-ok">matched</span>' if best[0] >= thr else '<span class="t-no">not matched</span>')
          if best[0] > 0 else '<span class="t-dim">no overlapping box</span>')
    return (t_head(f"{e(NAME[m])} · box #{k + 1} of {len(boxes)}") + t_loc(b, shape) + t_warn()
            + t_row("Element:", el_list([el])) + t_row(f"Best overlap in {NAME[o]}:", ov))


def mat_tip(m, x, shape):
    alts = "; ".join(f'{a.get("name", "")} ({a.get("note", "")})' if isinstance(a, dict) else str(a)
                     for a in x.get("alternatives") or [])
    return (t_head(f'{e(NAME[m])} · region {e(str(x.get("id", "")))}') + t_loc(x["box"], shape) + t_warn()
            + t_row("Elements:", el_list(x.get("elements")))
            + t_row("LLM Best Guess:", f'{e(x.get("material") or "")} '
                                       f'<span class="t-dim">(self-rated {e(str(x.get("confidence", "")))})</span>')
            + (t_row("Reasoning:", e(x["reasoning"])) if x.get("reasoning") else "")
            + (t_row("Alternatives:", e(alts)) if alts else ""))


def region_cell(x):
    if not x:
        return "<td class=mut>—</td>"
    mat = x.get("material") or ""
    return (f'<td><b>{"<i>unnamed</i>" if unnamed(mat) else e(mat)}</b> '
            f'<span class=mut>({e(str(x.get("confidence", "")))})</span>'
            f'<div class=mut>{el_list(x.get("elements"))}</div>'
            + (f'<div class=rsn>{e(x["reasoning"])}</div>' if x.get("reasoning") else "") + "</td>")


def build(run_dir: pathlib.Path) -> pathlib.Path:
    r = json.loads((run_dir / "results.json").read_text())
    s_name = next(iter(r["samples"]))
    smp = r["samples"][s_name]
    els = smp["elements"]
    thr = r.get("match_iou", 0.25)
    pthr = r.get("phase_match_overlap", 0.5)
    calls = [c for c in r.get("calls", []) if c.get("method") in (RAW, CAR) or c.get("kind") == "colocate"]
    cost = sum(c.get("cost_usd") or 0 for c in r.get("calls", []))
    bad = sum(1 for c in r.get("calls", []) if c.get("error") or c.get("truncated") or c.get("parse_status") == "failed")

    # ── Section 1: detection ──
    rows, chart, cards = [], [], []
    tot = {"raw": 0, "car": 0, "m": 0, "iw": 0.0}
    for en, el in els.items():
        br, bc = el["boxes"].get(RAW, []), el["boxes"].get(CAR, [])
        a = el.get("agreement", {}).get(f"{RAW}|{CAR}") or el.get("agreement", {}).get(f"{CAR}|{RAW}") or \
            {"matched": 0, "f1": 0.0, "mean_iou": 0.0}
        tot["raw"] += len(br); tot["car"] += len(bc); tot["m"] += a["matched"]; tot["iw"] += a["matched"] * a["mean_iou"]
        rows.append((el_list([en]), [[len(br), len(bc)], [a["matched"], f'{a["f1"]:.2f}', f'{a["mean_iou"]:.2f}']]))
        chart.append((en, {"raw": len(br), "car": len(bc)}))
        shape = el["shape"]
        base = el["images"].get(RAW, {}).get("original") or el["images"].get(CAR, {}).get("original", "")
        flags = "".join(f'<div class=warn>⚠ {e(NAME[m])}: {e(v)}</div>'
                        for m, v in {**el.get("errors", {}), **el.get("flags", {})}.items() if m in (RAW, CAR))

        def pane(m, bx, o, ob):
            ov = svg_boxes({CLS[m]: [(b, det_tip(m, k, bx, o, ob, en, shape, thr)) for k, b in enumerate(bx)]})
            inp = el["images"].get(m, {}).get("input", "")
            return (f'<div class=pane><div class=ph><i class="sw s-{CLS[m]}"></i><b>{NAME[m]}</b> · {len(bx)} boxes'
                    + (f' · <a href="{e(inp)}" target=_blank>image the VLM saw ↗</a>' if inp else "")
                    + f'</div><div class=pic><img loading=lazy src="{e(base)}" alt="{e(en)}">{ov}</div></div>')
        cards.append(f'<details><summary>{el_list([en])} <span class=mut>— raw {len(br)} · car 1 px {len(bc)} · '
                     f'matched {a["matched"]} · F1 {a["f1"]:.2f}</span></summary>{flags}'
                     f'<div class=pair>{pane(RAW, br, CAR, bc)}{pane(CAR, bc, RAW, br)}</div></details>')
    tf1 = 2 * tot["m"] / (tot["raw"] + tot["car"]) if (tot["raw"] + tot["car"]) else 1.0
    tiou = tot["iw"] / tot["m"] if tot["m"] else 0.0
    table1 = grouped_table("Element", [("Boxes", "reg", [f'<i class="sw s-raw"></i>{NAME[RAW]}', f'<i class="sw s-car"></i>{NAME[CAR]}']),
                                       (f"Agreement (IoU ≥ {thr})", "loc", ["Matched", "F1", "Mean IoU"])],
                           rows, ("All elements", [[tot["raw"], tot["car"]], [tot["m"], f"{tf1:.2f}", f"{tiou:.2f}"]]))

    # ── Section 2: materials ──
    regs = {m: smp["phase"]["regions"].get(m, []) for m in (RAW, CAR)}
    shape0 = next(iter(els.values()))["shape"]
    match = _greedy(regs[RAW], regs[CAR], pthr)  # {raw index: (car index, overlap)}
    pair_rows = []
    for i, (j, sc) in sorted(match.items(), key=lambda t: regs[RAW][t[0]]["box"]["y1"]):
        pair_rows.append(f'<tr><td class=num>{sc:.2f}</td>{region_cell(regs[RAW][i])}{region_cell(regs[CAR][j])}</tr>')
    only = {RAW: [x for i, x in enumerate(regs[RAW]) if i not in match],
            CAR: [x for j, x in enumerate(regs[CAR]) if j not in {v[0] for v in match.values()}]}
    only_html = "".join(
        f'<details class=inner><summary><i class="sw s-{CLS[m]}"></i>Only {NAME[m]} — {len(only[m])} region(s)</summary>'
        f'<table class=tbl><tbody>' + "".join(f"<tr>{region_cell(x)}</tr>" for x in only[m]) + "</tbody></table></details>"
        for m in (RAW, CAR))
    comp = smp["phase"].get("images", {}).get(RAW, {}).get("plain") or \
        smp["phase"].get("images", {}).get(CAR, {}).get("plain", "")
    ov2 = svg_boxes({CLS[m]: [(x["box"], mat_tip(m, x, shape0)) for x in regs[m]] for m in (RAW, CAR)})
    errs = "".join(f'<div class=warn>⚠ {e(NAME[m])}: {e(v)}</div>'
                   for m, v in smp["phase"].get("errors", {}).items() if m in (RAW, CAR))
    table2 = grouped_table("", [("Material regions", "reg", [f'<i class="sw s-raw"></i>{NAME[RAW]}', f'<i class="sw s-car"></i>{NAME[CAR]}']),
                                (f"Same location (overlap ≥ {pthr:.0%} of smaller box)", "loc",
                                 ["Matched pairs", f"Only {NAME[RAW]}", f"Only {NAME[CAR]}"])],
                           [("Count", [[len(regs[RAW]), len(regs[CAR])], [len(match), len(only[RAW]), len(only[CAR])]])])
    mi = r.get("model_input", {})

    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>VLM raw vs Car 1 px — {e(r.get("run", run_dir.name))}</title>
<style>
.viz-root{{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#dcdbd6;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--text-muted:#77766f;--grid:#e6e5e0;--s-raw:#6250d6;--s-car:#eb6834;--warn:#8a5a00;--warn-bg:#fff4dc}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9b9a90;--grid:#2c2b28;--s-raw:#9085e9;--s-car:#d95926;--warn:#f0c060;--warn-bg:#3a2e12}}}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,sans-serif}}
.viz-root{{background:var(--surface-0);color:var(--text-primary);min-height:100vh;padding:24px 32px 64px}}
h1{{margin:0 0 4px;font-size:22px}} h2{{margin:32px 0 6px;font-size:18px}} h4{{margin:0 0 6px}}
.mut{{color:var(--text-muted);font-weight:normal}} p.lead{{color:var(--text-secondary);max-width:950px}}
.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}
.tile{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:10px 16px;min-width:140px}}
.tile b{{display:block;font-size:22px}} .tile span{{color:var(--text-secondary);font-size:12px}}
.panel,.tw{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:10px 0}}
.tw{{padding:0;overflow-x:auto}} .tw table.grp{{width:100%;font-size:13px}}
.legend{{display:flex;gap:16px;color:var(--text-secondary);font-size:13px;margin-bottom:6px}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.sw.s-raw{{background:var(--s-raw)}} .sw.s-car{{background:var(--s-car)}}
.chart{{width:100%;height:auto;display:block}} .chart .grid{{stroke:var(--grid)}} .chart .ax,.chart .val{{fill:var(--text-secondary);font-size:11px}}
.bar{{stroke:var(--surface-1);stroke-width:1}} .bar.s-raw{{fill:var(--s-raw)}} .bar.s-car{{fill:var(--s-car)}}
details{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;margin:8px 0;padding:8px 12px}}
details>summary{{cursor:pointer;font-weight:600}} details.inner{{background:var(--surface-0)}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}}
.ph{{font-size:13px;margin-bottom:6px}} .ph a{{color:var(--text-secondary)}}
.pic{{position:relative;line-height:0}} .pic img{{width:100%;image-rendering:pixelated}}
.ov{{position:absolute;inset:0;width:100%;height:100%}}
.ov rect{{fill:transparent;stroke-width:2;vector-effect:non-scaling-stroke}} .ov rect:hover{{stroke-width:4}}
.g-raw rect{{stroke:var(--s-raw)}} .g-car rect{{stroke:var(--s-car);stroke-dasharray:4 2}}
body.hide-raw .g-raw,body.hide-car .g-car{{display:none}}
.mat{{display:grid;grid-template-columns:minmax(300px,520px) 1fr;gap:16px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface-1)}}
.tbl th,.tbl td{{border-bottom:1px solid var(--border);padding:6px 8px;text-align:left;vertical-align:top}}
.tbl th{{color:var(--text-secondary)}} td.num{{text-align:center;white-space:nowrap}}
.rsn{{color:var(--text-secondary);font-size:12px;margin-top:3px}}
.warn{{background:var(--warn-bg);color:var(--warn);border-radius:4px;padding:6px 10px;font-size:12.5px;margin:6px 0}}
.toggles{{position:sticky;top:0;z-index:5;background:var(--surface-0);padding:8px 0;border-bottom:1px solid var(--border);
 display:flex;gap:18px;font-size:13px;color:var(--text-secondary)}}
{TIP_CSS}
{GRP_CSS}
</style></head>
<body><div class=viz-root>
<h1>VLM raw image vs Car 1 px <span class=mut>— {e(s_name)}, {len(els)} elements</span></h1>
<p class=lead>Run <code>{e(r.get("run", run_dir.name))}</code> · {e(r.get("created", ""))} · model <code>{e(r.get("vlm_model") or "")}</code>
· sample context: <i>{e(r.get("config", {}).get("vlm", {}).get("sample_context", ""))}</i><br>
<b>{NAME[RAW]}</b>: {e(mi.get(RAW, ""))}.<br><b>{NAME[CAR]}</b>: {e(mi.get(CAR, ""))}.<br>
Neither uses OpenCV detection: the VLM finds the particles in both.</p>
<div class=tiles><div class=tile><b>{len(r.get("calls", []))}</b><span>API calls</span></div>
<div class=tile><b>${cost:.2f}</b><span>estimated cost</span></div>
<div class=tile><b>{bad}</b><span>failed / cut-off / unparsed replies</span></div></div>
<div class=toggles>Show boxes:
<label><input type=checkbox checked data-cls=raw> <i class="sw s-raw"></i>{NAME[RAW]}</label>
<label><input type=checkbox checked data-cls=car> <i class="sw s-car"></i>{NAME[CAR]}</label></div>

<h2>1 · Detection — boxes per element</h2>
<p class=lead>A box pair is matched when IoU ≥ {thr}. F1 = 2 × matched / (raw boxes + car boxes) — agreement between
the two, not accuracy. Both panes show the same normalized scan with that method's boxes; "image the VLM saw" opens its
actual input (the 150 px raw scan, or the 1024 px car mosaic).</p>
<div class=tiles><div class=tile><b>{tot["raw"]}</b><span>{NAME[RAW]} boxes</span></div>
<div class=tile><b>{tot["car"]}</b><span>{NAME[CAR]} boxes</span></div>
<div class=tile><b>{tot["m"]}</b><span>matched pairs</span></div>
<div class=tile><b>{tf1:.2f}</b><span>F1 agreement</span></div><div class=tile><b>{tiou:.2f}</b><span>mean IoU of matches</span></div></div>
<div class=panel><h4>Boxes per element</h4>
<div class=legend><span><i class="sw s-raw"></i>{NAME[RAW]}</span><span><i class="sw s-car"></i>{NAME[CAR]}</span></div>
{bar_chart(chart, [("raw", NAME[RAW]), ("car", NAME[CAR])], max([max(v.values()) for _, v in chart] + [1]))}</div>
<div class=tw>{table1}</div>
<h4>Side by side per element <span class=mut>— raw solid, car dashed; hover a box</span></h4>
{"".join(cards)}

<h2>2 · Materials — LLM material regions</h2>
<p class=lead>For each method, one LLM call reads that method's boxes for all {len(els)} elements and names the
multi-element regions (same co-location prompt for both). Two regions are a location match when they overlap by
≥ {pthr:.0%} of the smaller box.</p>
<div class=warn>⚠ LLM results may be inaccurate — they need manual review and correction.</div>
{errs}<div class=tw>{table2}</div>
<div class=mat><div class=panel><h4>Regions on the element composite <span class=mut>— hover a box</span></h4>
<div class=pic><img loading=lazy src="{e(comp)}" alt="composite">{ov2}</div></div>
<div><div class=panel><h4>Same location, side by side ({len(match)})</h4>
<table class=tbl><thead><tr><th>Overlap</th><th><i class="sw s-raw"></i>{NAME[RAW]}</th><th><i class="sw s-car"></i>{NAME[CAR]}</th></tr></thead>
<tbody>{"".join(pair_rows) or '<tr><td colspan=3 class=mut>No regions at the same location.</td></tr>'}</tbody></table></div>
{only_html}</div></div>
<div id=tip></div>
</div>
<script>
document.querySelectorAll('.toggles input').forEach(cb=>cb.addEventListener('change',()=>
  document.body.classList.toggle('hide-'+cb.dataset.cls,!cb.checked)));
const tip=document.getElementById('tip');
document.addEventListener('mousemove',ev=>{{const t=ev.target.closest('[data-tip]');
  if(!t){{tip.style.display='none';return}} tip.innerHTML=t.dataset.tip;tip.style.display='block';
  const r=tip.getBoundingClientRect();
  tip.style.left=Math.min(ev.clientX+14,innerWidth-r.width-8)+'px';tip.style.top=Math.min(ev.clientY+14,innerHeight-r.height-8)+'px';}});
</script></body></html>"""
    out = run_dir / "raw_vs_car1px.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    print("Report →", build(ap.parse_args().run_dir.resolve()))
