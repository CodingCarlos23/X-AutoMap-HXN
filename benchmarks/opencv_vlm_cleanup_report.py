#!/usr/bin/env python3
"""
OpenCV + Library + VLM cleanup report — built only from a run's results.json (no API calls, rerun freely).

  Panel 1  Initial OpenCV boxes (every element)      Panel 2  Material-library regions (R#)
  Panel 3  VLM-cleaned boxes (B#) + undetected particles (U#)

Usage:
  pixi run python benchmarks/opencv_vlm_cleanup_report.py benchmarks/runs/<timestamp>_opencv_vlm_cleanup
Writes <run>/opencv_vlm_cleanup.html.
"""
import argparse
import html
import json
import pathlib

from report import GRP_CSS, TIP_CSS, TIP_WARN_LIB, element_color, grouped_table
from car_vs_opencv_report import el_list, svg_boxes, t_head, t_loc, t_row, t_warn

OC = "opencv:simple:standard"
e = html.escape


def _name(n):
    return "<i>no library match</i>" if (not n or str(n).lower().startswith("no library match")) else e(n)


def _px(b, shape):
    h, w = shape
    return f'x {b["x1"] * w:.0f}–{b["x2"] * w:.0f}, y {b["y1"] * h:.0f}–{b["y2"] * h:.0f}'


def sample_html(s, sd, run_dir):
    c = sd.get("vlm_cleanup")
    if not c:
        return f"<h2>{e(s)}</h2><div class=warn>Step 3 (VLM cleanup) was not run for this sample.</div>"
    shape = next(iter(sd["elements"].values()))["shape"]
    st = c["stats"]
    comp = c["files"]["composite"]

    # panel 1: initial per-element boxes, coloured by element
    init_groups = {}
    for el, d in sd["elements"].items():
        boxes = d["boxes"].get(OC, [])
        init_groups[f"el{el}"] = [(b, t_head(f"OpenCV · {e(el)} box #{k + 1} of {len(boxes)}") + t_loc(b, shape)
                                   + t_row("Element:", el_list([el]))) for k, b in enumerate(boxes)]
    el_css = "".join(f".g-el{el} rect{{stroke:{element_color(el)}}}" for el in sd["elements"])
    p1 = svg_boxes(init_groups)

    # panel 2: library regions
    lib = []
    for r in c["regions"]:
        ratios = ", ".join(f"{el_list([k])} {v}%" for k, v in sorted(r["ratios"].items(), key=lambda t: -t[1]))
        lib.append((r["box"], t_head(f'Library · region {e(r["rid"])}') + t_loc(r["box"], shape)
                    + f'<div class="t-warn">{e(TIP_WARN_LIB)}</div>'
                    + t_row("Elements:", el_list(r["elements"])) + t_row("Library:", _name(r["material"]))
                    + (t_row("Ratios:", ratios) if ratios else "")
                    + t_row("Union id:", e(str(r["union_id"])))))
    p2 = svg_boxes({"lib": lib})

    # panel 3: VLM boxes + undetected
    vb = []
    for b in c["boxes"]:
        names = "".join(f'<div>{e(m["region"])}: {_name(m["name"])} <span class="t-dim">({el_list(m["elements"])})</span></div>'
                        for m in b["library"])
        vb.append((b["box"], t_head(f'VLM cleanup · box {e(b["id"])}') + t_loc(b["box"], shape) + t_warn()
                   + t_row("Merged from:", ", ".join(e(i) for i in b["merged_from"]))
                   + t_row("Elements:", el_list(b["elements"])) + t_row("Library names (unchanged):", names)
                   + (t_row("Note:", e(b["note"])) if b.get("note") else "")))
    ub = [(u["box"], t_head(e(u["id"])) + t_loc(u["box"], shape) + t_warn()
           + t_row("Description:", e(u["description"]))) for u in c["undetected"]]
    p3 = svg_boxes({"vlm": vb, "und": ub})

    def panel(n, title, sub, ov, extra=""):
        return (f'<div class=pane><div class=ph><b>{n} · {title}</b><div class=mut>{sub}</div></div>'
                f'<div class=pic><img loading=lazy src="{e(comp)}" alt="composite">{ov}</div>{extra}</div>')

    legend1 = '<div class=elleg>' + "".join(f'<span><i style="background:{element_color(el)}"></i>{e(el)}</span>'
                                            for el in sd["elements"]) + "</div>"
    panels = (panel(1, "Initial OpenCV detections", f'{c["initial_boxes"]} boxes, all {len(sd["elements"])} elements', p1, legend1)
              + panel(2, "Material library regions", f'{st["library_regions"]} regions · {st["overlap_pairs_library"]} overlapping pairs', p2)
              + panel(3, "VLM cleanup (new)", f'{st["vlm_boxes"]} boxes · {st["overlap_pairs_vlm"]} overlapping pairs · '
                      f'{st["undetected"]} undetected (dashed)', p3))

    stats = grouped_table("Stage", [("Boxes", "reg", ["Count", "Overlapping pairs"])], [
        ("1 · Initial OpenCV", [[c["initial_boxes"], "—"]]),
        ("2 · Material library", [[st["library_regions"], st["overlap_pairs_library"]]]),
        ("3 · VLM cleanup", [[st["vlm_boxes"], st["overlap_pairs_vlm"]]])])

    rows = []
    for b in c["boxes"]:
        lib_names = "".join(f'<div><b>{e(m["region"])}</b> {_name(m["name"])} <span class=mut>· {el_list(m["elements"])}</span></div>'
                            for m in b["library"])
        tag = "" if b["source"] == "vlm" else ' <span class=mut>(kept — not returned by VLM)</span>'
        rows.append(f'<tr><td><b>{e(b["id"])}</b>{tag}</td><td>{e(_px(b["box"], shape))}</td>'
                    f'<td>{len(b["merged_from"])}</td><td>{lib_names}</td><td>{el_list(b["elements"])}</td>'
                    f'<td class=mut>{e(b.get("note", ""))}</td></tr>')
    urows = "".join(f'<tr><td><b>{e(u["id"])}</b></td><td>{e(_px(u["box"], shape))}</td><td>{e(u["description"])}</td></tr>'
                    for u in c["undetected"]) or '<tr><td colspan=3 class=mut>None reported.</td></tr>'
    err = f'<div class=warn>⚠ VLM call failed: {e(c["error"])} — showing the library regions unchanged.</div>' if c.get("error") else ""
    dry = '<div class=warn>DRY RUN — no API call was made; panel 3 just repeats the library regions.</div>' if c.get("dry_run") else ""
    return f"""<h2>{e(s)}</h2>{err}{dry}
<div class=tiles>
<div class=tile><b>{c["initial_boxes"]}</b><span>initial OpenCV boxes</span></div>
<div class=tile><b>{st["library_regions"]}</b><span>library regions</span></div>
<div class=tile><b>{st["vlm_boxes"]}</b><span>VLM boxes</span></div>
<div class=tile><b>{st["overlap_pairs_library"]} → {st["overlap_pairs_vlm"]}</b><span>overlapping pairs (library → VLM)</span></div>
<div class=tile><b>{st["regions_merged"]} → {st["merged_boxes"]}</b><span>regions merged → into boxes</span></div>
<div class=tile><b>{st["undetected"]}</b><span>undetected particles</span></div></div>
<style>{el_css}</style>
<div class=panels>{panels}</div>
<div class=warn>⚠ Panel 3 is an LLM result — it may be inaccurate and needs manual review and correction.
Material names are the library's, unchanged; the VLM only reorganized locations.</div>
{f'<p class=lead><b>VLM summary:</b> {e(c["summary"])}</p>' if c.get("summary") else ""}
<div class=tw>{stats}</div>
<h3>VLM boxes ({len(c["boxes"])})</h3>
<table class=tbl><thead><tr><th>Box</th><th>Location (px)</th><th>Regions</th><th>Library regions inside (names unchanged)</th>
<th>Elements</th><th>VLM note</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h3>Undetected particles ({len(c["undetected"])})</h3>
<table class=tbl><thead><tr><th>Id</th><th>Location (px)</th><th>VLM description</th></tr></thead><tbody>{urows}</tbody></table>
<p class=lead>Files: <code>{e(s)}/vlm_cleanup/</code> — <code>cleanup.json</code>, <code>prompt.txt</code>, <code>reply.txt</code>,
<code>vlm_input.png</code> (what the VLM saw), <code>1_initial_boxes.png</code>, <code>2_library_boxes.png</code>, <code>3_vlm_boxes.png</code>.</p>"""


def build(run_dir: pathlib.Path) -> pathlib.Path:
    r = json.loads((run_dir / "results.json").read_text())
    calls = [c for c in r.get("calls", []) if c.get("kind") == "vlm_cleanup"]
    cost = sum(c.get("cost_usd") or 0 for c in calls)
    model = next((sd["vlm_cleanup"]["model"] for sd in r["samples"].values() if sd.get("vlm_cleanup")), "?")
    body = "".join(sample_html(s, sd, run_dir) for s, sd in r["samples"].items())
    flow = "".join(f'<div class="fs{" new" if n == 3 else ""}"><b>{n} · {t}</b><span>{d}</span></div>'
                   + ('<div class=arr>→</div>' if n < 4 else "") for n, t, d in [
                       (1, "OpenCV detection", "standard (dilated) blob boxes, per element"),
                       (2, "Material library", "multi-element unions + library names (deterministic)"),
                       (3, "VLM cleanup", "composite + library boxes + JSON → merge, tighten, find missing"),
                       (4, "Output", "composite with new boxes + cleanup.json")])
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>OpenCV + Library + VLM cleanup — {e(r.get("run", run_dir.name))}</title>
<style>
.viz-root{{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#dcdbd6;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--text-muted:#77766f;--warn:#8a5a00;--warn-bg:#fff4dc}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9b9a90;--warn:#f0c060;--warn-bg:#3a2e12}}}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,sans-serif}}
.viz-root{{background:var(--surface-0);color:var(--text-primary);min-height:100vh;padding:24px 32px 64px}}
h1{{margin:0 0 4px;font-size:22px}} h2{{margin:28px 0 6px;font-size:18px}} h3{{margin:22px 0 6px;font-size:15px}}
.mut{{color:var(--text-muted);font-weight:normal}} p.lead{{color:var(--text-secondary);max-width:1000px}}
.flow{{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:14px 0}}
.fs{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:8px 12px;max-width:230px}}
.fs b{{display:block}} .fs span{{font-size:12px;color:var(--text-secondary)}} .fs.new{{border:2px solid #1baf7a}}
.arr{{align-self:center;font-size:20px;color:var(--text-muted)}}
.tiles{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.tile{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:8px 14px;min-width:130px}}
.tile b{{display:block;font-size:20px}} .tile span{{color:var(--text-secondary);font-size:12px}}
.panels{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.pane{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:10px}}
.ph{{margin-bottom:6px}} .pic{{position:relative;line-height:0}} .pic img{{width:100%;image-rendering:pixelated}}
.ov{{position:absolute;inset:0;width:100%;height:100%}}
.ov rect{{fill:transparent;stroke-width:1.5;vector-effect:non-scaling-stroke}} .ov rect:hover{{stroke-width:3.5;fill:rgba(255,255,255,.15)}}
.g-lib rect{{stroke:#ffdc00}} .g-vlm rect{{stroke:#3ce65a;stroke-width:2}} .g-und rect{{stroke:#ff50dc;stroke-width:2;stroke-dasharray:5 3}}
.elleg{{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:11.5px;margin-top:6px}}
.elleg i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:3px}}
.tw{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;overflow-x:auto;margin:10px 0;max-width:520px}}
.tw table.grp{{width:100%;font-size:13px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface-1)}}
.tbl th,.tbl td{{border-bottom:1px solid var(--border);padding:6px 8px;text-align:left;vertical-align:top}}
.tbl th{{color:var(--text-secondary)}}
.warn{{background:var(--warn-bg);color:var(--warn);border-radius:4px;padding:6px 10px;font-size:12.5px;margin:8px 0}}
@media (max-width:1100px){{.panels{{grid-template-columns:1fr}}}}
{TIP_CSS}
{GRP_CSS}
</style></head>
<body><div class=viz-root>
<h1>OpenCV + Library + VLM cleanup</h1>
<p class=lead>Run <code>{e(r.get("run", run_dir.name))}</code> · {e(r.get("created", ""))} · VLM <code>{e(model)}</code> ·
{len(calls)} VLM call(s) · ${cost:.2f}. A new step after the deterministic material library: the VLM sees the
all-element composite with the library boxes and their JSON, merges boxes that mark the same particle, tightens them,
and lists particles nobody boxed. Library names are never changed. Workflow page: <code>docs/opencv_vlm_cleanup.html</code> ·
<a href="index.html">OpenCV / material-library details</a> · <a href="../index.html">all runs</a>.</p>
<div class=flow>{flow}</div>
{body}
<div id=tip></div></div>
<script>
const tip=document.getElementById('tip');
document.addEventListener('mousemove',ev=>{{const t=ev.target.closest('[data-tip]');
  if(!t){{tip.style.display='none';return}} tip.innerHTML=t.dataset.tip;tip.style.display='block';
  const r=tip.getBoundingClientRect();
  tip.style.left=Math.min(ev.clientX+14,innerWidth-r.width-8)+'px';tip.style.top=Math.min(ev.clientY+14,innerHeight-r.height-8)+'px';}});
</script></body></html>"""
    out = run_dir / "opencv_vlm_cleanup.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    print("Report →", build(ap.parse_args().run_dir.resolve()))
