#!/usr/bin/env python3
"""
Car Standalone vs OpenCV report — built only from a run's results.json (no API calls, rerun freely).

  Section 1  Detected boxes:  vlm:car_standalone  vs  opencv:simple:standard
  Section 2  Materials:       Car Standalone + LLM  vs  OpenCV + LLM  vs  OpenCV + materials library

Usage:
  pixi run python benchmarks/car_vs_opencv_report.py benchmarks/runs/<timestamp>_car_vs_opencv_llm
Writes <run>/car_vs_opencv.html (images are referenced relative to the run folder).
"""
import argparse
import html
import json
import pathlib
import re

OC, CAR, LLM = "opencv:simple:standard", "vlm:car_standalone", "opencv:simple:standard:llm"
NAMES = {OC: "OpenCV", CAR: "Car Standalone", LLM: "OpenCV + LLM"}
MAT_NAMES = {CAR: "Car VLM + LLM", LLM: "OpenCV + LLM", OC: "OpenCV + Library"}
CODE_M = {"L": OC, "O": LLM, "C": CAR}  # Level 3 codes → method ids
CLS = {OC: "oc", CAR: "car", LLM: "llm"}
MAT_ORDER = [CAR, LLM, OC]
L3_KINDS = [("all3", "All 3"), ("L|O", "Library + OpenCV LLM only"), ("L|C", "Library + Car only"), ("O|C", "OpenCV LLM + Car only")]
L3_MATCH = [("name_and_reasoning", "Name + reasoning"), ("name_only", "Name only"),
            ("reasoning_only", "Reasoning only"), ("none", "No material match")]

from report import (TIP_CSS, TIP_WARN, TIP_WARN_LIB, GRP_CSS, METHOD_SHORT, element_color, grouped_table,
                    l3_heatmap, method_key, method_swatch)  # same tooltip look + element colours as index.html

e = html.escape


# ── tooltip HTML (classes styled by TIP_CSS; stored escaped in data-tip, shown with innerHTML) ──

def t_head(h): return f'<div class="t-head">{h}</div>'
def t_row(lbl, h): return f'<div class="t-row"><span class="t-lbl">{e(lbl)}</span> {h}</div>'
def t_warn(lib=False): return f'<div class="t-warn">{e(TIP_WARN_LIB if lib else TIP_WARN)}</div>'
def el_span(x): return f'<span class="el" style="color:{element_color(x)}">{e(x)}</span>'
def el_list(xs): return ", ".join(el_span(x) for x in xs or [])


def t_loc(b, shape):
    h, w = shape
    x1, x2, y1, y2 = b["x1"] * w, b["x2"] * w, b["y1"] * h, b["y2"] * h
    return (f'<div class="t-loc">x {x1:.0f}–{x2:.0f}, y {y1:.0f}–{y2:.0f} px</div>'
            f'<div class="t-meta">center ({(x1 + x2) / 2:.0f}, {(y1 + y2) / 2:.0f}) · '
            f'{x2 - x1:.0f}×{y2 - y1:.0f} px · image {w}×{h}</div>')


def _iou(a, b):
    w = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    h = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - i)


def det_tip(m, k, boxes, other_m, other_boxes, el, shape, thr):
    """Section 1 box: title + id, location, disclaimer (Car), element, best overlap in the other method."""
    b = boxes[k]
    best = max(((_iou(b, o), j) for j, o in enumerate(other_boxes)), default=(0.0, None))
    ov = (f'box #{best[1] + 1}, IoU {best[0]:.2f} — '
          + ('<span class="t-ok">matched</span>' if best[0] >= thr else '<span class="t-no">not matched</span>')
          if best[0] > 0 else '<span class="t-dim">no overlapping box</span>')
    return (t_head(f"{e(NAMES[m])} · box #{k + 1} of {len(boxes)}") + t_loc(b, shape)
            + (t_warn() if m == CAR else "") + t_row("Element:", el_span(el))
            + t_row(f"Best overlap in {NAMES[other_m]}:", ov))


def mat_tip(m, x, shape):
    """Section 2 region: title + id, location, disclaimer (LLM methods), elements, name, ratios, LLM text."""
    mat = x.get("material") or ""
    if m == OC:
        name = (t_row("Library:", '<span class="t-dim">no library match</span>') if unnamed(mat)
                else t_row("Library match:", f'{e(mat)} <span class="t-dim">({e(str(x.get("confidence", "")))})</span>'))
    else:
        name = t_row("LLM Best Guess:", f'{e(mat)} <span class="t-dim">(self-rated {e(str(x.get("confidence", "")))})</span>')
    ratios = ", ".join(f"{el_span(k)} {v}%" for k, v in sorted((x.get("ratios") or {}).items(), key=lambda t: -t[1]))
    alts = "; ".join(f'{a.get("name", "")} ({a.get("note", "")})' if isinstance(a, dict) else str(a)
                     for a in x.get("alternatives") or [])
    return (t_head(f'{e(MAT_NAMES[m])} · region {e(str(x.get("id", "")))}') + t_loc(x["box"], shape)
            + t_warn(lib=m == OC) + t_row("Elements:", el_list(x.get("elements")))
            + name + (t_row("Ratios:", ratios) if ratios else "")
            + (t_row("Reasoning:", e(x["reasoning"])) if x.get("reasoning") else "")
            + (t_row("Alternatives:", e(alts)) if alts else ""))


def sample_key(name):
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 1e9, name)


def unnamed(name):
    return not name or str(name).lower().startswith("no library match")


def svg_boxes(boxes_by_cls, tip_prefix=""):
    """Vector overlay in a 0–1 viewBox; one <g> per method class so CSS toggles can hide it."""
    parts = ['<svg class="ov" viewBox="0 0 1 1" preserveAspectRatio="none">']
    for cls, boxes in boxes_by_cls.items():
        parts.append(f'<g class="g-{cls}">')
        for b, tip in boxes:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            parts.append(f'<rect x="{x1:.4f}" y="{y1:.4f}" width="{max(x2 - x1, .004):.4f}" '
                         f'height="{max(y2 - y1, .004):.4f}" data-tip="{e(tip)}"/>')
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def agreement(el):
    a = el.get("agreement", {})
    return a.get(f"{OC}|{CAR}") or a.get(f"{CAR}|{OC}") or {"matched": 0, "f1": 0.0, "mean_iou": 0.0}


def bar_chart(rows, series, max_v, fmt="{:.0f}", height=180, label_w=0):
    """Grouped vertical bars. rows = [(label, {series_cls: value, ...})]; series = [(cls, name)]."""
    n, k = len(rows), len(series)
    W, H, pad_l, pad_b, pad_t = 760, height, 36, 24, 10
    gw = (W - pad_l) / max(n, 1)
    bw = min(22, (gw - 10) / k - 2)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img">']
    for t in range(5):  # recessive grid
        v = max_v * t / 4
        y = H - pad_b - (H - pad_b - pad_t) * t / 4
        out.append(f'<line class="grid" x1="{pad_l}" x2="{W}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="ax" x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt.format(v)}</text>')
    for i, (label, vals) in enumerate(rows):
        x0 = pad_l + i * gw + (gw - k * (bw + 2)) / 2
        for j, (cls, name) in enumerate(series):
            v = vals.get(cls, 0) or 0
            h = (H - pad_b - pad_t) * (v / max_v if max_v else 0)
            x, y = x0 + j * (bw + 2), H - pad_b - h
            out.append(f'<path class="bar s-{cls}" d="M{x:.1f},{H - pad_b} V{y + min(4, h):.1f} '
                       f'q0,-{min(4, h):.1f} {min(4, h):.1f},-{min(4, h):.1f} h{bw - 2 * min(4, h):.1f} '
                       f'q{min(4, h):.1f},0 {min(4, h):.1f},{min(4, h):.1f} V{H - pad_b} Z" '
                       f'data-tip="{e(t_head(e(label)) + t_row(name + ":", fmt.format(v)))}"/>')
        out.append(f'<text class="ax" x="{pad_l + i * gw + gw / 2:.1f}" y="{H - 6}" text-anchor="middle">{e(label)}</text>')
    out.append("</svg>")
    return "".join(out)


def hbar_chart(rows, max_v, cls="neutral"):
    """Single-series horizontal bars: rows = [(label, value, tip)]."""
    W, rh, lw = 760, 26, 170
    H = rh * len(rows) + 6
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img">']
    for i, (label, v, tip) in enumerate(rows):
        y = 3 + i * rh
        w = (W - lw - 50) * (v / max_v if max_v else 0)
        out.append(f'<text class="ax" x="{lw - 8}" y="{y + 16}" text-anchor="end">{e(label)}</text>')
        if w > 0:
            r = min(4, w)
            out.append(f'<path class="bar s-{cls}" d="M{lw},{y + 4} H{lw + w - r:.1f} q{r:.1f},0 {r:.1f},{r:.1f} '
                       f'V{y + 18 - r:.1f} q0,{r:.1f} -{r:.1f},{r:.1f} H{lw} Z" data-tip="{e(tip)}"/>')
        out.append(f'<text class="val" x="{lw + w + 6:.1f}" y="{y + 16}">{v}</text>')
    out.append("</svg>")
    return "".join(out)


def build(run_dir: pathlib.Path) -> pathlib.Path:
    r = json.loads((run_dir / "results.json").read_text())
    snames = sorted(r["samples"], key=sample_key)
    label = {s: f"Sample {i + 1}" for i, s in enumerate(snames)}

    calls = r.get("calls", [])
    cost = sum(c.get("cost_usd") or 0 for c in calls)
    n_err = sum(1 for c in calls if c.get("error"))
    n_bad = sum(1 for c in calls if c.get("truncated") or c.get("parse_status") == "failed")

    # ── Section 1 data ──
    s1_rows, s1_chart, tot = [], [], {"oc": 0, "car": 0, "m": 0, "iou_w": 0.0}
    s1_detail = []
    for s in snames:
        els = r["samples"][s]["elements"]
        n_oc = sum(len(el["boxes"].get(OC, [])) for el in els.values())
        n_car = sum(len(el["boxes"].get(CAR, [])) for el in els.values())
        m = sum(agreement(el)["matched"] for el in els.values())
        iou_w = sum(agreement(el)["matched"] * agreement(el)["mean_iou"] for el in els.values())
        f1 = 2 * m / (n_oc + n_car) if (n_oc + n_car) else 1.0
        tot["oc"] += n_oc; tot["car"] += n_car; tot["m"] += m; tot["iou_w"] += iou_w
        s1_chart.append((label[s], {"oc": n_oc, "car": n_car}))
        s1_rows.append((f"{label[s]} <span class=mut>{e(s)}</span>",
                        [[len(els)], [n_oc, n_car], [m, f"{f1:.2f}", f"{(iou_w / m) if m else 0:.2f}"]]))

        cards = []
        for en, el in els.items():
            a = agreement(el)
            img = el["images"].get(OC, {}).get("original") or el["images"].get(CAR, {}).get("original", "")
            mosaic = el["images"].get(CAR, {}).get("input_boxes", "")
            err = el.get("errors", {}).get(CAR)
            flag = el.get("flags", {}).get(CAR)
            bo, bc, shp, thr = el["boxes"].get(OC, []), el["boxes"].get(CAR, []), el["shape"], r.get("match_iou", 0.25)
            ov = svg_boxes({
                "oc": [(b, det_tip(OC, k, bo, CAR, bc, en, shp, thr)) for k, b in enumerate(bo)],
                "car": [(b, det_tip(CAR, k, bc, OC, bo, en, shp, thr)) for k, b in enumerate(bc)],
            })
            note = (f'<div class=warn>⚠ Car: {e(err)}</div>' if err else "") + \
                   (f'<div class=warn>⚠ Car: {e(flag)}</div>' if flag else "")
            cards.append(
                f'<div class=card><div class=ct><b>{e(en)}</b>'
                f'<span><i class="sw s-oc"></i>{len(el["boxes"].get(OC, []))} '
                f'<i class="sw s-car"></i>{len(el["boxes"].get(CAR, []))} · '
                f'matched {a["matched"]} · F1 {a["f1"]:.2f}</span></div>'
                f'<div class=pic><img loading=lazy src="{e(img)}" alt="{e(en)}">{ov}</div>{note}'
                + (f'<a class=lnk href="{e(mosaic)}" target=_blank>car mosaic the VLM saw ↗</a>' if mosaic else "")
                + "</div>")
        s1_detail.append(f'<details><summary>{label[s]} <span class=mut>{e(s)} · {len(els)} elements</span>'
                         f'</summary><div class=grid>{"".join(cards)}</div></details>')
    tot_f1 = 2 * tot["m"] / (tot["oc"] + tot["car"]) if (tot["oc"] + tot["car"]) else 1.0
    tot_iou = tot["iou_w"] / tot["m"] if tot["m"] else 0.0

    # ── Section 2 data (Level 3: 3A location agreement + 3B material agreement) ──
    kind_tot = {k: 0 for k, _ in L3_KINDS}
    only_tot = {c: 0 for c in "LOC"}
    reg_tot = {c: 0 for c in "LOC"}
    match_tot = {m: 0 for m, _ in L3_MATCH}
    who_tot = {m: {k: 0 for k, _ in L3_KINDS} for m, _ in L3_MATCH}
    n_sets = n_judged = 0
    s2a_rows, s2b_rows, s2_chart, s2_detail = [], [], [], []
    for s in snames:
        smp = r["samples"][s]
        regs = smp["phase"]["regions"]
        l3 = smp.get("level3")
        counts = {m: len(regs.get(m, [])) for m in MAT_ORDER}
        s2_chart.append((label[s], {CLS[m]: counts[m] for m in MAT_ORDER}))
        set_rows = []
        if l3:
            c3 = l3["counts"]
            k3 = {"all3": c3["all3"], **c3["pairs"]}
            for k, _ in L3_KINDS:
                kind_tot[k] += k3[k]
            for c in "LOC":
                only_tot[c] += c3["only"][c]; reg_tot[c] += c3["regions"][c]
            mc = l3.get("material_counts") or {}
            per = {m: sum((mc.get(m) or {}).values()) for m, _ in L3_MATCH}
            for m, _ in L3_MATCH:
                match_tot[m] += per[m]
                for k, _ in L3_KINDS:
                    who_tot[m][k] += (mc.get(m) or {}).get(k, 0)
            n_sets += len(l3["sets"]); n_judged += l3.get("n_judged", 0)
            s2a_rows.append((f"{label[s]} <span class=mut>{e(s)}</span>",
                             [[c3["regions"][c] for c in "LOC"], [k3[k] for k, _ in L3_KINDS],
                              [c3["only"][c] for c in "LOC"]]))
            s2b_rows.append((f"{label[s]} <span class=mut>{e(s)}</span>",
                             [[f"{l3.get('n_judged', 0)} / {len(l3['sets'])}"], [per[m] for m, _ in L3_MATCH]]))
            for st in l3["sets"]:
                cells = []
                for c in "LOC":
                    mbs = st["members"].get(c) or []
                    if not mbs:
                        cells.append("<td class=miss>—</td>")
                        continue
                    cells.append("<td>" + '<hr style="border:0;border-top:1px dashed var(--border);margin:4px 0">'.join(
                        f'<i class="sw s-{CLS[CODE_M[c]]}"></i>'
                        + ("<i>unnamed</i>" if unnamed(mb["material"]) else e(mb["material"]))
                        + f'<div class=mut>{el_list(mb["elements"])}</div>' for mb in mbs) + "</td>")
                who = st.get("agree") or []
                who_s = "All 3" if len(who) == 3 else " + ".join(MAT_NAMES[CODE_M[c]] for c in who) if who else "—"
                ov = ", ".join(f"{v:.2f}" for v in st["overlaps"].values())
                set_rows.append(f'<tr><td><b>{e(st["id"])}</b></td><td>{e(dict(L3_KINDS)[st["kind"]])}'
                                f'{" <span class=mut>(chained)</span>" if st.get("chained") else ""}'
                                f'<div class=mut>overlap {ov}</div></td>{"".join(cells)}'
                                f'<td><b>{e(dict(L3_MATCH).get(st.get("match"), "not judged"))}</b>'
                                f'<div class=mut>{e(who_s)}</div></td><td class=mut>{e(st.get("note") or "")}</td></tr>')

        comp = smp["phase"].get("images", {}).get(OC, {}).get("plain", "")
        shp = next(iter(smp["elements"].values()))["shape"]
        ov = svg_boxes({CLS[m]: [(x["box"], mat_tip(m, x, shp)) for x in regs.get(m, [])] for m in MAT_ORDER})
        lists = []
        for m in MAT_ORDER:
            items = []
            for k, x in enumerate(regs.get(m, []), 1):
                alts = "; ".join(f'{a.get("name", "")} ({a.get("note", "")})' if isinstance(a, dict) else str(a)
                                 for a in x.get("alternatives", []) or [])
                mat = x.get("material") or ""
                items.append(f'<li><b>{"<i>unnamed</i>" if unnamed(mat) else e(mat)}</b> '
                             f'<span class=mut>[{e(", ".join(x.get("elements", [])))}] · {e(str(x.get("confidence", "")))}'
                             f'</span>' + (f'<div>{e(x["reasoning"])}</div>' if x.get("reasoning") else "")
                             + (f'<div class=mut>Alternatives: {e(alts)}</div>' if alts else "") + "</li>")
            lists.append(f'<details class=inner><summary><i class="sw s-{CLS[m]}"></i>{MAT_NAMES[m]} — '
                         f'{len(items)} region(s)</summary><ol>{"".join(items)}</ol></details>')
        err = smp["phase"].get("errors", {})
        errs = "".join(f'<div class=warn>⚠ {e(MAT_NAMES.get(m, m))}: {e(v)}</div>'
                       for m, v in err.items() if m in MAT_ORDER)
        j_err = ((l3 or {}).get("judge") or {}).get("error")
        if j_err:
            errs += f'<div class=warn>⚠ Level 3 material judge: {e(j_err)}</div>'
        if not l3:
            errs += '<div class=warn>Level 3 not computed for this run.</div>'
        s2_detail.append(
            f'<details><summary>{label[s]} <span class=mut>{e(s)} · {len(set_rows)} location sets · '
            f'{(l3 or {}).get("counts", {}).get("all3", 0)} found by all 3</span></summary>'
            f'{errs}<div class=mat><div class="pic big"><img loading=lazy src="{e(comp)}" alt="composite">{ov}</div>'
            f'<div class=matside><h4>Material regions per method</h4>{"".join(lists)}</div></div>'
            f'<table class=tbl><thead><tr><th>Set</th><th>Found by</th>'
            + "".join(f"<th>{MAT_NAMES[CODE_M[c]]}</th>" for c in "LOC")
            + f'<th>Material match</th><th>Note</th></tr></thead><tbody>{"".join(set_rows)}</tbody></table></details>')

    loc_rows = ([(n, kind_tot[k], t_head(e(n)) + t_row("Location sets (all samples):", str(kind_tot[k])))
                 for k, n in L3_KINDS]
                + [(f"{MAT_NAMES[CODE_M[c]]} only", only_tot[c],
                    t_head(e(MAT_NAMES[CODE_M[c]]) + " only") + t_row("Regions no other method found:", str(only_tot[c])))
                   for c in "LOC"])
    mat_rows = [(n, match_tot[m], t_head(e(n)) + t_row("Matched sets (all samples):", str(match_tot[m])))
                for m, n in L3_MATCH]
    mname = lambda c: f"{method_swatch(c)}{METHOD_SHORT[c]}"
    short = METHOD_SHORT
    kname = lambda k: "All 3" if k == "all3" else " + ".join(method_swatch(c) + short[c] for c in k.split("|")) + " only"
    table_s1 = grouped_table("Sample", [("", "info", ["Elements"]),
                                        ("Boxes", "reg", [f'{method_swatch("L")}OpenCV', f'{method_swatch("C")}Car Standalone']),
                                        ("Agreement (IoU ≥ " + str(r.get("match_iou", 0.25)) + ")", "loc",
                                         ["Matched", "F1", "Mean IoU"])],
                             s1_rows, ("All samples", [[""], [tot["oc"], tot["car"]],
                                                       [tot["m"], f"{tot_f1:.2f}", f"{tot_iou:.2f}"]]))
    table_2a = grouped_table("Sample", [("Material regions", "reg", [mname(c) for c in "LOC"]),
                                        ("Same location found by (each spot counted once)", "loc", [kname(k) for k, _ in L3_KINDS]),
                                        ("Found by this method only", "only", [mname(c) for c in "LOC"])],
                             s2a_rows, ("All samples", [[reg_tot[c] for c in "LOC"], [kind_tot[k] for k, _ in L3_KINDS],
                                                        [only_tot[c] for c in "LOC"]]))
    table_who = grouped_table("Material match", [("Methods that agreed", "who", [kname(k) for k, _ in L3_KINDS]),
                                                 ("", "info", ["Total"])],
                              [(e(n), [[who_tot[m][k] for k, _ in L3_KINDS], [f"<b>{match_tot[m]}</b>"]])
                               for m, n in L3_MATCH])
    table_2b = grouped_table("Sample", [("Judge", "info", ["Sets judged"]),
                                        ("Material match", "match", [e(n) for _, n in L3_MATCH])],
                             s2b_rows, ("All samples", [[f"{n_judged} / {n_sets}"], [match_tot[m] for m, _ in L3_MATCH]]))
    max_box = max([max(v.values()) for _, v in s1_chart] + [1])
    max_reg = max([max(v.values()) for _, v in s2_chart] + [1])
    thr = r.get("phase_match_overlap", 0.5)
    judge_model = next((r["samples"][s]["level3"].get("judge", {}).get("model") for s in snames
                        if r["samples"][s].get("level3")), "?")

    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>Car Standalone vs OpenCV — {e(r.get("run", run_dir.name))}</title>
<style>
.viz-root{{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#dcdbd6;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--text-muted:#77766f;--grid:#e6e5e0;--s-oc:#2a78d6;--s-car:#eb6834;--s-llm:#1baf7a;
 --s-neutral:#8a8984;--warn:#8a5a00;--warn-bg:#fff4dc}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9b9a90;--grid:#2c2b28;--s-oc:#3987e5;--s-car:#d95926;--s-llm:#199e70;--s-neutral:#77766f;
 --warn:#f0c060;--warn-bg:#3a2e12}}}}
*{{box-sizing:border-box}}
body{{margin:0;font:14px/1.45 system-ui,sans-serif}}
.viz-root{{background:var(--surface-0);color:var(--text-primary);min-height:100vh;padding:24px 32px 64px}}
h1{{margin:0 0 4px;font-size:22px}} h2{{margin:36px 0 6px;font-size:18px}} h4{{margin:0 0 6px}}
.mut{{color:var(--text-muted);font-weight:normal}} p.lead{{color:var(--text-secondary);max-width:900px}}
.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.tile{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:10px 16px;min-width:140px}}
.tile b{{display:block;font-size:22px}} .tile span{{color:var(--text-secondary);font-size:12px}}
.panel{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin:12px 0}}
.legend{{display:flex;gap:16px;color:var(--text-secondary);font-size:13px;margin-bottom:6px;flex-wrap:wrap}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.sw.s-oc{{background:var(--s-oc)}} .sw.s-car{{background:var(--s-car)}} .sw.s-llm{{background:var(--s-llm)}}
.tbl{{border-collapse:collapse;width:100%;background:var(--surface-1);font-size:13px;margin:8px 0}}
.tbl th,.tbl td{{border-bottom:1px solid var(--border);padding:6px 8px;text-align:left;vertical-align:top}}
.tbl th{{color:var(--text-secondary);font-weight:600}} .tbl tfoot td{{font-weight:600}}
td.miss{{color:var(--text-muted)}}
.chart{{width:100%;height:auto;display:block}} .chart .grid{{stroke:var(--grid);stroke-width:1}}
.chart .ax,.chart .val{{fill:var(--text-secondary);font-size:11px}}
.bar{{stroke:var(--surface-1);stroke-width:1}} .bar:hover{{opacity:.8}}
.bar.s-oc{{fill:var(--s-oc)}} .bar.s-car{{fill:var(--s-car)}} .bar.s-llm{{fill:var(--s-llm)}} .bar.s-neutral{{fill:var(--s-neutral)}}
details{{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;margin:8px 0;padding:8px 12px}}
details>summary{{cursor:pointer;font-weight:600}} details.inner{{padding:4px 8px;margin:6px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;margin-top:10px}}
.card{{border:1px solid var(--border);border-radius:6px;padding:8px;background:var(--surface-0)}}
.ct{{display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;gap:6px}}
.ct span{{color:var(--text-secondary)}}
.pic{{position:relative;line-height:0}} .pic img{{width:100%;image-rendering:pixelated}}
.ov{{position:absolute;inset:0;width:100%;height:100%}}
.ov rect{{fill:transparent;stroke-width:2;vector-effect:non-scaling-stroke}}
.ov rect:hover{{stroke-width:4;fill:rgba(255,255,255,.12)}}
.g-oc rect{{stroke:var(--s-oc)}} .g-car rect{{stroke:var(--s-car)}} .g-llm rect{{stroke:var(--s-llm)}}
.ov .g-car rect{{stroke-dasharray:4 2}}
body.hide-oc .g-oc,body.hide-car .g-car,body.hide-llm .g-llm{{display:none}}
.lnk{{font-size:12px;color:var(--text-secondary)}}
.warn{{background:var(--warn-bg);color:var(--warn);border-radius:4px;padding:4px 8px;font-size:12px;margin:4px 0}}
.mat{{display:grid;grid-template-columns:minmax(280px,480px) 1fr;gap:16px;margin-top:10px}}
.matside p{{color:var(--text-secondary);margin-top:0}} .matside ol{{padding-left:20px;font-size:13px}}
.matside li{{margin-bottom:6px}}
.toggles{{position:sticky;top:0;z-index:5;background:var(--surface-0);padding:8px 0;border-bottom:1px solid var(--border);
 display:flex;gap:18px;font-size:13px;color:var(--text-secondary)}}
{TIP_CSS}
{GRP_CSS}
.tw{{overflow-x:auto;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;margin:8px 0}}
.tw table.grp{{width:100%;font-size:13px}}
</style></head>
<body><div class=viz-root>
<h1>Car Standalone vs OpenCV <span class=mut>— {len(snames)} samples</span></h1>
<p class=lead>Run <code>{e(r.get("run", run_dir.name))}</code> · {e(r.get("created", ""))} · model
<code>{e(r.get("vlm_model", ""))}</code> · sample context: <i>{e(r.get("config", {}).get("vlm", {}).get("sample_context", ""))}</i><br>
OpenCV = <code>{OC}</code> ({e(r.get("model_input", {}).get(OC, ""))}).
Car Standalone = <code>{CAR}</code> ({e(r.get("model_input", {}).get(CAR, ""))}).</p>
<div class=tiles>
<div class=tile><b>{len(calls)}</b><span>API calls</span></div>
<div class=tile><b>${cost:.2f}</b><span>estimated cost</span></div>
<div class=tile><b>{n_err}</b><span>failed calls</span></div>
<div class=tile><b>{n_bad}</b><span>truncated / unparsed replies</span></div>
</div>
<div class=toggles>Show boxes:
<label><input type=checkbox checked data-cls=oc> <i class="sw s-oc"></i>OpenCV / library</label>
<label><input type=checkbox checked data-cls=car> <i class="sw s-car"></i>Car Standalone</label>
<label><input type=checkbox checked data-cls=llm> <i class="sw s-llm"></i>OpenCV + LLM</label></div>

<h2>1 · Detected boxes</h2>
<p class=lead>Every element of every sample. A box pair is matched when IoU ≥ {r.get("match_iou", 0.25)}.
F1 = 2 × matched / (OpenCV boxes + Car boxes); it measures agreement between the two methods, not accuracy
against ground truth.</p>
<div class=tiles>
<div class=tile><b>{tot["oc"]}</b><span>OpenCV boxes</span></div>
<div class=tile><b>{tot["car"]}</b><span>Car Standalone boxes</span></div>
<div class=tile><b>{tot["m"]}</b><span>matched pairs</span></div>
<div class=tile><b>{tot_f1:.2f}</b><span>overall F1</span></div>
<div class=tile><b>{tot_iou:.2f}</b><span>mean IoU of matches</span></div>
</div>
<div class=panel><h4>Boxes per sample (all elements)</h4>
<div class=legend><span><i class="sw s-oc"></i>OpenCV</span><span><i class="sw s-car"></i>Car Standalone</span></div>
{bar_chart(s1_chart, [("oc", "OpenCV"), ("car", "Car Standalone")], max_box)}</div>
<div class=tw>{table_s1}</div>
<h4>Per element <span class=mut>— OpenCV solid, Car dashed, on the normalized scan; hover a box</span></h4>
{"".join(s1_detail)}

<h2>2 · Materials</h2>
<p class=lead>Three material methods, all on the Level 2 multi-element regions:
<b>OpenCV + Library</b> (deterministic ratio match), <b>OpenCV + LLM</b> (an LLM names OpenCV's regions) and
<b>Car VLM + LLM</b> (an LLM names the Car VLM's regions).</p>
<div class=legend>{"".join(f'<span><i class="sw s-{CLS[m]}"></i>{MAT_NAMES[m]}</span>' for m in [OC, LLM, CAR])}</div>
<div class=panel><h4>Material regions per sample</h4>
{bar_chart(s2_chart, [(CLS[m], MAT_NAMES[m]) for m in MAT_ORDER], max_reg)}</div>

<h3>2A · Location agreement — same spot found by which methods</h3>
<p class=lead>Code, no LLM: two regions match when their boxes overlap by at least {thr:.0%} of the smaller box,
one-to-one, and chained: regions connected through matches form one set (one spot), so every region is counted once.
<b>Same location found by</b> counts sets; <b>Found by this method only</b> counts single regions nobody else found.</p>
<div class=tiles>
<div class=tile><b>{kind_tot["all3"]}</b><span>found by all 3</span></div>
<div class=tile><b>{kind_tot["L|O"] + kind_tot["L|C"] + kind_tot["O|C"]}</b><span>found by 2</span></div>
<div class=tile><b>{sum(only_tot.values())}</b><span>found by 1 only</span></div>
</div>
<div class=panel><h4>Who found the same location (all samples)</h4>{hbar_chart(loc_rows, max([v for _, v, _ in loc_rows] + [1]))}</div>
{method_key()}<div class=tw>{table_2a}</div>
<div class=panel><h4>Location agreement heatmap</h4>
<p class=lead style="margin-top:0">Each cell = number of spots in that sample that <b>both</b> methods found — including
spots the third method also found (also counted under All 3), so a pair column is always ≥ All 3. Darker = more spots.
The table above counts each spot once (its pair columns exclude the All 3 spots).</p>
{l3_heatmap({label[s]: r["samples"][s]["level3"] for s in snames if r["samples"][s].get("level3")}, "car")}</div>

<h3>2B · Material agreement — did the matched methods name the same material?</h3>
<div class=warn>⚠ Judged by an LLM ({e(str(judge_model))}, one call per sample) — may be inaccurate; check the notes.
The library has no written reasoning, so its elements and ratios stand in for reasoning; a library region with no
library match only counts as a location match.</div>
<p class=lead><b>Name + reasoning</b> = same material and same explanation · <b>Name only</b> = same material
(wording may differ), different explanation · <b>Reasoning only</b> = different names, same chemistry/source ·
<b>No material match</b> = same spot, different material.</p>
<div class=panel><h4>Material agreement on the {n_sets} location sets (all samples)</h4>{hbar_chart(mat_rows, max([v for _, v, _ in mat_rows] + [1]))}</div>
<div class=tw>{table_who}</div>
<p class=lead>Columns = which methods agreed. For <i>No material match</i> they show where the regions were found.</p>
<div class=tw>{table_2b}</div>
<h4>Per sample <span class=mut>— regions on the element composite (hover a box) and every location set with its verdict</span></h4>
{"".join(s2_detail)}
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
    out = run_dir / "car_vs_opencv.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    print("Report →", build(ap.parse_args().run_dir.resolve()))
