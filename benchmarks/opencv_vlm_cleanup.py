#!/usr/bin/env python3
"""
OpenCV + Library + VLM cleanup — benchmark for a new step after the deterministic material library.

  1. OpenCV standard detection        per-element blob boxes                       (AutoMap, unchanged)
  2. Material library                 multi-element union regions + library names   (AutoMap, unchanged)
  3. VLM cleanup (NEW)                the VLM sees the all-element composite with the library boxes drawn
                                      and numbered (R1, R2 …) + their JSON, and returns:
                                        - merged / tightened boxes with less overlap (each lists the R ids
                                          it came from — library names are kept, never renamed)
                                        - "Undetected particle #k": particles visible in the image with no box
  4. Output                           composite with the new boxes + cleanup.json per sample

Usage (from the repo root):
  pixi run python benchmarks/opencv_vlm_cleanup.py benchmarks/configs/opencv_vlm_cleanup.json        # full run
  pixi run python benchmarks/opencv_vlm_cleanup.py benchmarks/configs/opencv_vlm_cleanup.json --dry-run
                                                   # steps 1–2 + VLM input image/prompt, NO API call
  pixi run python benchmarks/opencv_vlm_cleanup.py --cleanup-only benchmarks/runs/<run> [--dry-run]
                                                   # redo step 3 on an existing run (keeps steps 1–2)
Then the report is written to <run>/opencv_vlm_cleanup.html (rebuild free: opencv_vlm_cleanup_report.py).
"""
import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

OC = "opencv:simple:standard"
VLM_SIZE = 1024  # composite sent to the VLM (nearest-neighbour upscale of the 150 px scan)

SYSTEM = (
    "You are an XRF (X-ray fluorescence) image analyst. A deterministic pipeline (OpenCV blob detection + a "
    "materials library) drew bounding boxes for multi-element particles on an all-element composite image. "
    "Its boxes often overlap: the same particle is reported several times by different element groups, or one "
    "particle is split into pieces. Your job is to clean the boxes up so each real particle has one good box, "
    "and to point out particles that have no box at all. You never rename materials. Return ONLY valid JSON."
)

RULES = """TASK
1. MERGE: boxes that mark the SAME particle (overlapping, nested or touching boxes on one blob) become ONE output
   box covering that particle. List the input ids it came from in "merged_from".
2. TIGHTEN: fit each output box to the particle you see in the image (exclude dark background). Output boxes
   should not overlap unless two distinct particles really touch.
3. KEEP: a box that is already fine stays as its own output box with "merged_from": [its id].
   Every input id (R1 … Rn) must appear in exactly one output box's "merged_from" — do not drop any.
4. Do NOT rename anything — material names stay as given; you only reorganize locations.
5. UNDETECTED: list bright particles clearly visible in the image that NO input box covers, as
   "Undetected particle #1", "#2", … with a box and a short description (colour / likely elements).
   Only clear particles — not noise. Use [] if there are none.
6. Coordinates are FRACTIONS of the image: x1, y1, x2, y2 between 0 and 1, (0,0) = top-left, x1 < x2, y1 < y2.
7. "note": ONE short sentence per output box (why merged / tightened / kept). "summary": 1–2 sentences overall."""

SCHEMA = ('{"boxes": [{"id": "B1", "x1": 0.10, "y1": 0.20, "x2": 0.18, "y2": 0.27, "merged_from": ["R3", "R7"], '
          '"note": "R3 and R7 mark the same Fe-rich particle; merged and tightened."}], '
          '"undetected": [{"id": "Undetected particle #1", "x1": 0.50, "y1": 0.60, "x2": 0.53, "y2": 0.64, '
          '"description": "small bright green spot, no box"}], "summary": "..."}')


# ── geometry ───────────────────────────────────────────────────────────────────

def _inter(a, b):
    w = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    h = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    return w * h if (w > 0 and h > 0) else 0.0


def overlap_pairs(boxes: list) -> int:
    """Number of box pairs whose areas intersect at all."""
    return sum(1 for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if _inter(boxes[i], boxes[j]) > 0)


def _clean_box(d: dict):
    try:
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(d[k]))) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


# ── images ─────────────────────────────────────────────────────────────────────

def composite_rgb(tiff_paths: list) -> np.ndarray:
    """Same all-element composite as before (feature_locator_headless._build_composite), at native size."""
    from automap_hxn.feature_locator_headless import _build_composite
    return _build_composite([str(p) for p in tiff_paths], normalize=True, upscale=False, circles=False)


def upscale(rgb: np.ndarray, size: int = VLM_SIZE) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    return cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_NEAREST)


def draw_boxes(rgb: np.ndarray, boxes: list, color, labels=None, thick=2, dashed=False) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    for k, b in enumerate(boxes):
        p1 = (int(b["x1"] * w), int(b["y1"] * h))
        p2 = (max(p1[0] + 1, int(b["x2"] * w)), max(p1[1] + 1, int(b["y2"] * h)))
        if dashed:
            for (a, c) in (((p1[0], p1[1]), (p2[0], p1[1])), ((p2[0], p1[1]), (p2[0], p2[1])),
                           ((p2[0], p2[1]), (p1[0], p2[1])), ((p1[0], p2[1]), (p1[0], p1[1]))):
                n = max(1, int(np.hypot(c[0] - a[0], c[1] - a[1]) // 8))
                for t in range(0, n, 2):
                    q1 = (int(a[0] + (c[0] - a[0]) * t / n), int(a[1] + (c[1] - a[1]) * t / n))
                    q2 = (int(a[0] + (c[0] - a[0]) * (t + 1) / n), int(a[1] + (c[1] - a[1]) * (t + 1) / n))
                    cv2.line(out, q1, q2, color, thick)
        else:
            cv2.rectangle(out, p1, p2, color, thick)
        if labels:
            org = (p1[0] + 2, max(12, p1[1] + 13))
            cv2.putText(out, labels[k], org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, labels[k], org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


def save_rgb(rgb: np.ndarray, path: pathlib.Path):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# ── prompt / reply ─────────────────────────────────────────────────────────────

def build_prompt(regions: list, elements: list, sample_context: str) -> str:
    lines = []
    if sample_context:
        lines += [f"Sample: {sample_context}.", ""]
    lines += [f"The image is an additive false-colour composite of {len(elements)} XRF element maps "
              f"({', '.join(elements)}). Colours are reused for some elements, so use the element lists below — "
              "not the colour — to know which elements a box contains.",
              "Yellow numbered boxes (R1 … Rn) are the deterministic material-library regions. Their data:", ""]
    for r in regions:
        b = r["box"]
        ratios = ", ".join(f"{k} {v}%" for k, v in sorted((r.get("ratios") or {}).items(), key=lambda t: -t[1])[:5])
        name = r["material"] if r["material"] and not r["material"].lower().startswith("no library match") \
            else "no library match"
        lines.append(f'{r["rid"]} | box {b["x1"]:.3f},{b["y1"]:.3f},{b["x2"]:.3f},{b["y2"]:.3f} | '
                     f'elements {",".join(r["elements"])} | library: {name}' + (f" | ratios {ratios}" if ratios else ""))
    lines += ["", RULES, "", f"Return JSON with this schema:\n{SCHEMA}"]
    return "\n".join(lines)


def parse_reply(text: str) -> dict:
    t = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    a, b = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[a:b + 1]) if 0 <= a < b else {}
    except json.JSONDecodeError:
        return {}


def postprocess(reply: dict, regions: list) -> dict:
    """VLM reply → final boxes (every R id in exactly one box) + undetected list + stats."""
    by_id = {r["rid"]: r for r in regions}
    seen, boxes = set(), []
    for k, ob in enumerate(reply.get("boxes", []) or [], 1):
        if not isinstance(ob, dict):
            continue
        box = _clean_box(ob)
        ids = [str(x) for x in ob.get("merged_from") or [] if str(x) in by_id and str(x) not in seen]
        if not box or not ids:
            continue
        seen.update(ids)
        boxes.append({"box": box, "merged_from": ids, "note": str(ob.get("note", "")), "source": "vlm"})
    for r in regions:  # anything the VLM left out stays as its own box, so nothing disappears
        if r["rid"] not in seen:
            boxes.append({"box": r["box"], "merged_from": [r["rid"]], "source": "kept",
                          "note": "not returned by the VLM — kept unchanged"})
    for k, bx in enumerate(boxes, 1):
        mem = [by_id[i] for i in bx["merged_from"]]
        bx["id"] = f"B{k}"
        bx["elements"] = sorted({e for m in mem for e in m["elements"]})
        bx["library"] = [{"region": m["rid"], "union_id": m["union_id"], "name": m["material"],
                          "confidence": m.get("confidence"), "elements": m["elements"]} for m in mem]
    undetected = []
    for k, u in enumerate(reply.get("undetected", []) or [], 1):
        box = _clean_box(u) if isinstance(u, dict) else None
        if box:
            undetected.append({"id": f"Undetected particle #{len(undetected) + 1}", "box": box,
                               "description": str(u.get("description", ""))})
    stats = {"library_regions": len(regions), "vlm_boxes": len(boxes),
             "merged_boxes": sum(len(b["merged_from"]) > 1 for b in boxes),
             "regions_merged": sum(len(b["merged_from"]) for b in boxes if len(b["merged_from"]) > 1),
             "kept_unreturned": sum(b["source"] == "kept" for b in boxes),
             "overlap_pairs_library": overlap_pairs([r["box"] for r in regions]),
             "overlap_pairs_vlm": overlap_pairs([b["box"] for b in boxes]),
             "undetected": len(undetected)}
    return {"boxes": boxes, "undetected": undetected, "summary": str(reply.get("summary", "")), "stats": stats}


# ── step 3 on one run ──────────────────────────────────────────────────────────

def cleanup_run(run_dir: pathlib.Path, dry_run: bool = False) -> dict:
    from automap_hxn import vlm_pipeline
    from run_comparison import DEFAULT_PRICING, call_record
    rp = run_dir / "results.json"
    results = json.loads(rp.read_text())
    cfg = results.get("config", {})
    ccfg = cfg.get("vlm_cleanup", {})
    model = ccfg.get("model") or cfg.get("vlm", {}).get("model", "claude-sonnet-4-6")
    ctx = cfg.get("vlm", {}).get("sample_context", "")
    pricing = {**DEFAULT_PRICING, **cfg.get("pricing", {})}
    provider = vlm_pipeline.get_provider(model)
    key = a_url = o_url = None
    if not dry_run:
        key, a_url, o_url = vlm_pipeline.resolve_key(provider)
        if not key and provider != "ollama":
            sys.exit("[ERROR] no API key found — load it first:\n"
                     '  eval "$(gpg --quiet --batch --yes --decrypt ~/.private_env.gpg)"\n'
                     f"then: pixi run python benchmarks/opencv_vlm_cleanup.py --cleanup-only {run_dir}")
    results["calls"] = [c for c in results.get("calls", []) if c.get("kind") != "vlm_cleanup"]
    for s, sd in results["samples"].items():
        out = run_dir / s / "vlm_cleanup"
        out.mkdir(parents=True, exist_ok=True)
        tdir = run_dir / s / "automap_opencv_simple_standard"
        elements = list(sd["elements"])
        tiffs = [tdir / sd["elements"][e]["file"] for e in elements]
        comp = composite_rgb(tiffs)
        big = upscale(comp)
        regs = sorted(sd["phase"]["regions"].get(OC, []), key=lambda r: (r["box"]["y1"], r["box"]["x1"]))
        regions = [{"rid": f"R{k}", "union_id": r.get("id"), "box": r["box"], "elements": r["elements"],
                    "material": r.get("material") or "", "confidence": r.get("confidence"),
                    "ratios": r.get("ratios") or {}} for k, r in enumerate(regs, 1)]
        init = [b for e in elements for b in sd["elements"][e]["boxes"].get(OC, [])]
        vlm_in = draw_boxes(big, [r["box"] for r in regions], (255, 220, 0), [r["rid"] for r in regions])
        save_rgb(big, out / "composite.png")
        save_rgb(vlm_in, out / "vlm_input.png")
        save_rgb(draw_boxes(big, init, (255, 255, 255), thick=1), out / "1_initial_boxes.png")
        save_rgb(draw_boxes(big, [r["box"] for r in regions], (255, 220, 0)), out / "2_library_boxes.png")
        prompt = build_prompt(regions, elements, ctx)
        (out / "prompt.txt").write_text(f"SYSTEM:\n{SYSTEM}\n\nUSER (+ image vlm_input.png):\n{prompt}")
        res = {"model": model, "elements": elements, "regions": regions, "initial_boxes": len(init),
               "error": None, "dry_run": dry_run, "files": {"composite": f"{s}/vlm_cleanup/composite.png",
                                                            "vlm_input": f"{s}/vlm_cleanup/vlm_input.png"}}
        if dry_run:
            res.update(postprocess({}, regions))
            res["summary"] = "DRY RUN — no API call; every library region kept as its own box."
            print(f"  [{s}] dry run: {len(regions)} library regions, {len(init)} initial boxes, prompt "
                  f"{len(prompt)} chars → {out}")
        else:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(vlm_in, cv2.COLOR_RGB2BGR))
            import base64
            meta, t0 = {}, time.time()
            try:
                text = vlm_pipeline.query_vlm_text(base64.b64encode(buf.tobytes()).decode(), SYSTEM, prompt, model,
                                                   provider, key or "", anthropic_url=a_url, openai_url=o_url,
                                                   max_tokens=int(ccfg.get("max_tokens", 16384)), meta=meta)
                (out / "reply.txt").write_text(text)
                reply = parse_reply(text)
                meta["parse_status"] = "ok" if reply.get("boxes") else "failed"
                res.update(postprocess(reply, regions))
                res["files"]["reply"] = f"{s}/vlm_cleanup/reply.txt"
            except Exception as ex:
                res["error"] = f"{type(ex).__name__}: {ex}"
                res.update(postprocess({}, regions))
            results["calls"].append(call_record("vlm_cleanup", "vlm_cleanup", None, meta, time.time() - t0, pricing,
                                                model, error=res["error"],
                                                raw_path=res["files"].get("reply")))
            st = res["stats"]
            print(f"  [{s}] {st['library_regions']} library regions → {st['vlm_boxes']} VLM boxes "
                  f"({st['regions_merged']} merged into {st['merged_boxes']}), overlapping pairs "
                  f"{st['overlap_pairs_library']} → {st['overlap_pairs_vlm']}, {st['undetected']} undetected"
                  + (f" · ERROR {res['error']}" if res["error"] else ""))
        save_rgb(draw_boxes(draw_boxes(big, [b["box"] for b in res["boxes"]], (60, 230, 90),
                                       [b["id"] for b in res["boxes"]]),
                            [u["box"] for u in res["undetected"]], (255, 80, 220),
                            [f"U{k}" for k in range(1, len(res["undetected"]) + 1)], dashed=True),
                 out / "3_vlm_boxes.png")
        (out / "cleanup.json").write_text(json.dumps(res, indent=2))
        sd["vlm_cleanup"] = res
    results["vlm_model"] = results.get("vlm_model") or model  # steps 1–2 ran without a VLM
    rp.write_text(json.dumps(results, indent=2))
    import opencv_vlm_cleanup_report
    from report import write_report, write_runs_index
    write_report(results, run_dir / "index.html")  # adds the banner pointing to the 3-panel page
    write_runs_index(run_dir.parent)                # runs list links this run to the 3-panel page
    print("Report →", opencv_vlm_cleanup_report.build(run_dir))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", type=pathlib.Path, help="benchmark config (runs steps 1–3)")
    ap.add_argument("--cleanup-only", type=pathlib.Path, help="existing run folder: redo only step 3")
    ap.add_argument("--dry-run", action="store_true", help="no API call (build images + prompt only)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None, help="runs folder (default benchmarks/runs/)")
    args = ap.parse_args()
    if args.cleanup_only:
        cleanup_run(args.cleanup_only.resolve(), args.dry_run)
        return
    if not args.config:
        ap.error("give a config, or --cleanup-only <run folder>")
    import run_comparison
    config = json.loads(args.config.read_text())
    print("Steps 1–2: OpenCV standard detection + material library (no API calls) …")
    run_dir = run_comparison.run(config, args.config.resolve(), no_vlm=True,
                                 runs_root=args.out_dir.resolve() if args.out_dir else None)
    print(f"\nStep 3: VLM cleanup{' (dry run)' if args.dry_run else ''} …")
    cleanup_run(run_dir, args.dry_run)


if __name__ == "__main__":
    main()
