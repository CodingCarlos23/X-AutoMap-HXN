#!/usr/bin/env python3
"""
Model sweep report — one VLM method, several models (only the model changes).
Built only from the sweep folder's saved results (no API calls, rerun freely).

Usage:
  pixi run python benchmarks/model_sweep_report.py benchmarks/runs/<timestamp>_<sweep name>
Writes <sweep>/model_sweep.html.
"""
import argparse
import html
import json
import pathlib

from report import GRP_CSS, TIP_CSS, grouped_table
from car_vs_opencv_report import el_list, svg_boxes, t_head, t_loc, t_row, t_warn, unnamed

e = html.escape
VENDOR = lambda m: "Claude" if m.startswith("claude") else "OpenAI" if m.startswith(("gpt", "o1", "o3", "o4")) else "Other"


def _iou(a, b):
    w = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    h = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - i)


def match(a, b, thr):
    """One-to-one greedy IoU matching → (matched, sum of IoU)."""
    pairs = sorted(((_iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    ua, ub, n, s = set(), set(), 0, 0.0
    for sc, i, j in pairs:
        if sc < thr:
            break
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); n += 1; s += sc
    return n, s


def hbars(rows, unit=""):
    """Horizontal bars, one per model, coloured by vendor: rows = [(label, value, vendor)]."""
    W, rh, lw = 760, 28, 170
    mx = max([v for _, v, _ in rows] + [1])
    H = rh * len(rows) + 6
    out = [f'<svg class=chart viewBox="0 0 {W} {H}" role="img">']
    for k, (lab, v, ven) in enumerate(rows):
        y = 3 + k * rh
        w = (W - lw - 70) * v / mx
        out.append(f'<text class="ax" x="{lw - 8}" y="{y + 17}" text-anchor="end">{e(lab)}</text>')
        if w > 0:
            out.append(f'<rect class="bar v-{ven}" x="{lw}" y="{y + 4}" width="{w:.1f}" height="18" rx="3" '
                       f'data-tip="{e(t_head(e(lab)) + t_row("Value:", f"{v:,.0f}{unit}"))}"/>')
        out.append(f'<text class="val" x="{lw + w + 6:.1f}" y="{y + 17}">{v:,.0f}{unit}</text>')
    out.append("</svg>")
    return "".join(out)


def build(sweep: pathlib.Path) -> pathlib.Path:
    meta = json.loads((sweep / "sweep.json").read_text())
    runs = {}
    for m in meta["models"]:
        if m.get("folder") and not m.get("excluded") and (sweep / m["folder"] / "results.json").exists():
            runs[m["model"]] = json.loads((sweep / m["folder"] / "results.json").read_text())
    models = [m["model"] for m in meta["models"] if m["model"] in runs]
    failed = [m for m in meta["models"] if m["model"] not in runs and not m.get("excluded")]
    excluded = [m for m in meta["models"] if m.get("excluded")]
    base = meta.get("baseline") if meta.get("baseline") in runs else (models[0] if models else None)
    any_r = runs[models[0]] if models else {}
    method = next(iter(any_r.get("vlm_variants") or []), "vlm:car_standalone")
    s_name = next(iter(any_r.get("samples", {"?": {}})))
    els = list(any_r["samples"][s_name]["elements"]) if models else []
    thr = any_r.get("match_iou", 0.25)
    mi = any_r.get("model_input", {}).get(method, "")

    def boxes(m, el):
        return runs[m]["samples"][s_name]["elements"].get(el, {}).get("boxes", {}).get(method, [])

    # ── per-model summary ──
    summ, chart_boxes, chart_time = [], [], []
    for m in models:
        r = runs[m]
        calls = r.get("calls", [])
        tb = sum(len(boxes(m, el)) for el in els)
        regs = r["samples"][s_name]["phase"]["regions"].get(method, [])
        tin = sum(c.get("input_tokens") or 0 for c in calls)
        tout = sum(c.get("output_tokens") or 0 for c in calls)
        loc_t = [c["time_s"] for c in calls if c.get("kind") == "locate" and c.get("time_s")]
        cost = sum(c.get("cost_usd") or 0 for c in calls)
        priced = all(c.get("cost_usd") is not None for c in calls if not c.get("error"))
        bad = sum(1 for c in calls if c.get("error") or c.get("truncated") or c.get("parse_status") == "failed")
        if base and m != base:
            mt = si = nb = 0
            for el in els:
                n, s = match(boxes(base, el), boxes(m, el), thr)
                mt += n; si += s; nb += len(boxes(base, el))
            f1 = 2 * mt / (nb + tb) if (nb + tb) else 0.0
            ag = [mt, f"{f1:.2f}", f"{si / mt if mt else 0:.2f}"]
        else:
            ag = ["baseline", "—", "—"]
        summ.append((f'<span class="vd v-{VENDOR(m)}"></span><b>{e(m)}</b>',
                     [[tb, len(regs)], [f"{sum(loc_t) / len(loc_t):.0f} s" if loc_t else "—", f"{tin:,}", f"{tout:,}",
                                         f"${cost:.2f}" if priced else "not priced"],
                      [len(calls), bad], ag]))
        chart_boxes.append((m, tb, VENDOR(m)))
        chart_time.append((m, sum(loc_t) / len(loc_t) if loc_t else 0, VENDOR(m)))
    t_sum = grouped_table("Model", [("Results", "reg", ["Boxes", "Material regions"]),
                                    ("Cost of the run", "info", ["Mean time / image call", "Input tokens",
                                                                 "Output tokens", "Cost"]),
                                    ("Replies", "only", ["Calls", "Failed / cut off / unparsed"]),
                                    (f"Agreement with {e(base or '?')} (IoU ≥ {thr})", "loc",
                                     ["Matched", "F1", "Mean IoU"])], summ)
    t_el = grouped_table("Element", [("Boxes found", "reg", [e(m) for m in models])],
                         [(f"<b>{e(el)}</b>", [[len(boxes(m, el)) for m in models]]) for el in els],
                         ("All elements", [[sum(len(boxes(m, el)) for el in els) for m in models]]))

    # ── per-element panels ──
    details = []
    for el in els:
        d0 = runs[base or models[0]]["samples"][s_name]["elements"][el]
        img = d0["images"].get(method, {}).get("original", "")
        shape = d0["shape"]
        panes = []
        for m in models:
            bx = boxes(m, el)
            folder = next(x["folder"] for x in meta["models"] if x["model"] == m)
            d = runs[m]["samples"][s_name]["elements"][el]
            err = d.get("errors", {}).get(method) or d.get("flags", {}).get(method)
            ov = svg_boxes({f"v{VENDOR(m)}": [(b, t_head(f"{e(m)} · box #{k + 1} of {len(bx)}") + t_loc(b, shape)
                                              + t_warn() + t_row("Element:", el_list([el]))) for k, b in enumerate(bx)]})
            panes.append(f'<div class=pane><div class=ph><span class="vd v-{VENDOR(m)}"></span><b>{e(m)}</b> · {len(bx)}'
                         + (f' <span class=bad title="{e(err)}">⚠</span>' if err else "")
                         + f'</div><div class=pic><img loading=lazy src="{e(folder)}/{e(img)}" alt="">{ov}</div></div>')
        details.append(f'<details><summary>{e(el)} <span class=mut>— '
                       + " · ".join(f"{e(m)} {len(boxes(m, el))}" for m in models)
                       + f'</span></summary><div class=gridm>{"".join(panes)}</div></details>')

    # ── materials ──
    mpanes, mlists = [], []
    shape0 = runs[models[0]]["samples"][s_name]["elements"][els[0]]["shape"] if models else [1, 1]
    for m in models:
        folder = next(x["folder"] for x in meta["models"] if x["model"] == m)
        ph = runs[m]["samples"][s_name]["phase"]
        regs = ph["regions"].get(method, [])
        comp = ph.get("images", {}).get(method, {}).get("plain", "")
        tips = [(x["box"], t_head(f'{e(m)} · region {e(str(x.get("id", "")))}') + t_loc(x["box"], shape0) + t_warn()
                 + t_row("Elements:", el_list(x.get("elements")))
                 + t_row("LLM Best Guess:", f'{e(x.get("material") or "")} <span class="t-dim">(self-rated '
                                            f'{e(str(x.get("confidence", "")))})</span>')
                 + (t_row("Reasoning:", e(x["reasoning"])) if x.get("reasoning") else "")) for x in regs]
        err = ph.get("errors", {}).get(method)
        mpanes.append(f'<div class=pane><div class=ph><span class="vd v-{VENDOR(m)}"></span><b>{e(m)}</b> · '
                      f'{len(regs)} regions' + (f' <span class=bad title="{e(err)}">⚠</span>' if err else "")
                      + f'</div><div class=pic><img loading=lazy src="{e(folder)}/{e(comp)}" alt="">'
                      f'{svg_boxes({f"v{VENDOR(m)}": tips})}</div></div>')
        mlists.append(f'<details><summary>{e(m)} — {len(regs)} material regions</summary><ol class=rl>' + "".join(
            f'<li><b>{"<i>unnamed</i>" if unnamed(x.get("material")) else e(x.get("material") or "")}</b> '
            f'<span class=mut>({e(str(x.get("confidence", "")))}) · {e(", ".join(x.get("elements") or []))}</span>'
            + (f'<div class=mut>{e(x["reasoning"])}</div>' if x.get("reasoning") else "") + "</li>" for x in regs)
            + "</ol></details>")

    fail_html = "".join(f'<div class=warn>{e(m["model"])} left out of this report: {e(m["excluded"])} '
                        f'(its files are kept in {e(str(m.get("folder")))}/).</div>' for m in excluded)
    fail_html += "".join(f'<div class=warn>⚠ {e(m["model"])}: no results (exit {m.get("exit_code")}) — see '
                        f'logs/{e(m["model"])}.log</div>' for m in failed)
    total_cost = sum(c.get("cost_usd") or 0 for r in runs.values() for c in r.get("calls", []))
    total_calls = sum(len(r.get("calls", [])) for r in runs.values())
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>Model sweep — {e(sweep.name)}</title>
<style>
.viz-root{{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#dcdbd6;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--text-muted:#77766f;--grid:#e6e5e0;--v-claude:#eb6834;--v-openai:#1baf7a;--v-other:#8a8984;
 --warn:#8a5a00;--warn-bg:#fff4dc}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9b9a90;--grid:#2c2b28;--v-claude:#d95926;--v-openai:#199e70;--warn:#f0c060;--warn-bg:#3a2e12}}}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,sans-serif}}
.viz-root{{background:var(--surface-0);color:var(--text-primary);min-height:100vh;padding:24px 32px 64px}}
h1{{margin:0 0 4px;font-size:22px}} h2{{margin:30px 0 6px;font-size:18px}} h4{{margin:0 0 6px}}
.mut{{color:var(--text-muted);font-weight:normal}} p.lead{{color:var(--text-secondary);max-width:1000px}}
.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}
.tile{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:8px 14px;min-width:130px}}
.tile b{{display:block;font-size:20px}} .tile span{{color:var(--text-secondary);font-size:12px}}
.panel{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:10px 0;max-width:820px}}
.legend{{display:flex;gap:16px;font-size:13px;color:var(--text-secondary);margin-bottom:6px}}
.vd{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.vd.v-Claude,.bar.v-Claude{{background:var(--v-claude);fill:var(--v-claude)}}
.vd.v-OpenAI,.bar.v-OpenAI{{background:var(--v-openai);fill:var(--v-openai)}}
.vd.v-Other,.bar.v-Other{{background:var(--v-other);fill:var(--v-other)}}
.tw{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;overflow-x:auto;margin:10px 0;width:fit-content;max-width:100%}}
.tw table.grp{{font-size:13px}}
.chart{{width:100%;height:auto;display:block}} .chart .ax,.chart .val{{fill:var(--text-secondary);font-size:12px}}
details{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;margin:8px 0;padding:8px 12px}}
details>summary{{cursor:pointer;font-weight:600}}
.gridm{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:10px}}
.pane{{background:var(--surface-0);border:1px solid var(--border);border-radius:6px;padding:6px}}
.ph{{font-size:12.5px;margin-bottom:4px}} .bad{{color:#c0392b}}
.pic{{position:relative;line-height:0}} .pic img{{width:100%;image-rendering:pixelated}}
.ov{{position:absolute;inset:0;width:100%;height:100%}}
.ov rect{{fill:transparent;stroke-width:1.6;vector-effect:non-scaling-stroke}} .ov rect:hover{{stroke-width:3.5}}
.g-vClaude rect{{stroke:var(--v-claude)}} .g-vOpenAI rect{{stroke:var(--v-openai)}} .g-vOther rect{{stroke:var(--v-other)}}
.rl{{font-size:13px;padding-left:22px}} .rl li{{margin-bottom:5px}}
.warn{{background:var(--warn-bg);color:var(--warn);border-radius:4px;padding:6px 10px;font-size:12.5px;margin:8px 0}}
{TIP_CSS}
{GRP_CSS}
</style></head>
<body><div class=viz-root>
<h1>Model sweep <span class=mut>— {e(method)} · {e(s_name)}, {len(els)} elements</span></h1>
<p class=lead>Sweep <code>{e(sweep.name)}</code> · {e(meta.get("created", ""))}. Only the model changes: same images,
prompt and settings for every model ({e(mi)}). The same model also names the materials. Baseline for agreement:
<b>{e(base or "?")}</b>. Agreement is not accuracy: there is no hand-labelled ground truth.</p>
<div class=legend><span><i class="vd v-Claude"></i>Claude</span><span><i class="vd v-OpenAI"></i>OpenAI</span></div>
<div class=tiles><div class=tile><b>{len(models)}</b><span>models with results</span></div>
<div class=tile><b>{total_calls}</b><span>API calls</span></div>
<div class=tile><b>${total_cost:.2f}</b><span>priced cost (some models not priced)</span></div></div>
{fail_html}

<h2>1 · Summary per model</h2>
<div class=tw>{t_sum}</div>
<div class=panel><h4>Boxes found (all elements)</h4>{hbars(chart_boxes)}</div>
<div class=panel><h4>Mean time per image call</h4>{hbars(chart_time, " s")}</div>

<h2>2 · Detection per element</h2>
<div class=tw>{t_el}</div>
<p class=lead>Every panel is the same normalized scan with that model's boxes; the input image (car mosaic) is identical for all models.</p>
{"".join(details)}

<h2>3 · Materials</h2>
<div class=warn>⚠ LLM results may be inaccurate — they need manual review and correction.</div>
<div class=gridm>{"".join(mpanes)}</div>
{"".join(mlists)}
<div id=tip></div></div>
<script>
const tip=document.getElementById('tip');
document.addEventListener('mousemove',ev=>{{const t=ev.target.closest('[data-tip]');
  if(!t){{tip.style.display='none';return}} tip.innerHTML=t.dataset.tip;tip.style.display='block';
  const r=tip.getBoundingClientRect();
  tip.style.left=Math.min(ev.clientX+14,innerWidth-r.width-8)+'px';tip.style.top=Math.min(ev.clientY+14,innerHeight-r.height-8)+'px';}});
</script></body></html>"""
    out = sweep / "model_sweep.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=pathlib.Path)
    print("Report →", build(ap.parse_args().sweep_dir.resolve()))
