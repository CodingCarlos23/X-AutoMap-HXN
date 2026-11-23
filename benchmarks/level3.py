#!/usr/bin/env python3
"""
Level 3 — agreement between the three material methods, built only from the Level 2 material regions.

  L = OpenCV + Library   (deterministic ratio match; no written reasoning — its evidence is elements + ratios)
  O = OpenCV + LLM       (LLM names OpenCV's multi-element regions)
  C = Car VLM + LLM      (LLM names the Car Standalone VLM's multi-element regions)

3A  Location agreement (code, no LLM)
    Regions of two methods match when their boxes overlap by ≥ `overlap_thresh` of the smaller box
    (same rule as Level 2), one-to-one per method pair, best overlap first. Matches are then chained: every
    group of regions connected through matches is ONE set (so A–B and A–C make one 3-method set even if B and C
    do not touch). A set can hold more than one region of a method (e.g. the library split one particle in two).
    Every region lands in exactly one set; a region matched to nothing is "only".

3B  Material agreement (one LLM call per sample, only on the 3A sets)
    The LLM judges each set: did the methods name the same material (wording may differ) and/or give the same
    reasoning (same chemistry / source)? → name_and_reasoning | name_only | reasoning_only | none, plus which
    methods agree and a one-sentence note. A library region with no library match never counts as a material
    match (only as a location match).

Apply to an existing run (adds "level3" to results.json and rebuilds that run's pages):
  pixi run python benchmarks/level3.py benchmarks/runs/<run>
"""
import argparse
import concurrent.futures
import json
import pathlib
import shutil
import sys
import time

CODES = ("L", "O", "C")
CODE_NAMES = {"L": "OpenCV + Library", "O": "OpenCV + LLM", "C": "Car VLM + LLM"}
PAIRS = (("L", "O"), ("L", "C"), ("O", "C"))
MATCH_TYPES = ("name_and_reasoning", "name_only", "reasoning_only", "none")
MATCH_LABELS = {"name_and_reasoning": "Name + reasoning", "name_only": "Name only",
                "reasoning_only": "Reasoning only", "none": "No material match"}

SYSTEM = (
    "You are an XRF (X-ray fluorescence) materials scientist checking whether different analysis methods "
    "identified the SAME material for the SAME particle. The regions in each set were already matched by "
    "location; you only judge the material identification. Be strict but chemistry-aware. Be concise. "
    "Return ONLY valid JSON."
)

RULES = """RULES
1. NAME match: the names refer to the same material, compound, mineral or phase, even if worded differently
   (e.g. hematite = Fe2O3 = iron oxide; "Fe-rich steel particle" ≈ "stainless steel"). A general name matches a
   specific one only if the specific one is a kind of it (e.g. "Fe oxide" vs "hematite" = match;
   "iron oxide" vs "steel" = no match; "gypsum" vs "calcium sulfate" = match).
2. REASONING match: the explanations point to the same chemistry and source (same dominant elements, same
   interpretation). Method L has no written reasoning: compare its element list and ratios with the others'
   reasoning (e.g. L ratios dominated by Fe + Cr + Ni and an LLM reasoning about stainless steel = match).
3. A method-L region named "unnamed (no library match)" can NOT match on name or reasoning — judge only the others.
   If a method has several regions in one set, judge its identification as a whole (any of its regions may match).
4. For each set give:
   "match": "name_and_reasoning" | "name_only" | "reasoning_only" | "none" — the strongest agreement found;
   "agree": EVERY method code whose identification fits that agreement (at least 2; [] when "none").
            If all three methods fit, list all three — do not narrow it to the best-matching pair. List only two
            when the third clearly names or explains a different material (or is an unnamed library region);
   "note": ONE short sentence — what matched or differed (e.g. "O and C both call it stainless steel; L only
            matches by Fe-Cr ratios").
5. Every set ID must appear exactly once."""

SCHEMA = ('{"sets": [{"set": "S1", "match": "name_only", "agree": ["O", "C"], '
          '"note": "Both name gypsum; reasoning differs (O: construction dust, C: secondary sulfate)."}]}')


# ── 3A: location agreement ────────────────────────────────────────────────────

def _overlap(a: dict, b: dict) -> float:
    """Intersection ÷ area of the smaller box (fractional 0–1 boxes)."""
    w = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    h = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a["x2"] - a["x1"]) * (a["y2"] - a["y1"]), (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return (w * h) / small if small > 0 else 0.0


def _greedy(a: list, b: list, thresh: float) -> dict:
    """One-to-one matches, best overlap first → {i: (j, score)}."""
    pairs = sorted(((_overlap(x["box"], y["box"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)),
                   reverse=True)
    used_a, used_b, out = set(), set(), {}
    for s, i, j in pairs:
        if s < thresh:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out[i] = (j, s)
    return out


def _member(code: str, i: int, r: dict) -> dict:
    return {"index": i, "id": r.get("id"), "box": r["box"], "elements": r.get("elements", []),
            "material": r.get("material") or "", "confidence": r.get("confidence"),
            "reasoning": r.get("reasoning") or "", "alternatives": r.get("alternatives") or [],
            "ratios": r.get("ratios") or {}}


def unnamed(name) -> bool:
    return not name or str(name).lower().startswith("no library match")


def location_sets(regions: dict, thresh: float) -> dict:
    """regions = {"L": [...], "O": [...], "C": [...]} (Level 2 regions). Returns sets + 3A counts.

    Pairwise one-to-one matches are the edges; each connected group of regions is one set, so every region is
    counted exactly once. kind = "all3", "L|O", "L|C", "O|C" (methods present) or "only:<code>" (unmatched)."""
    m = {f"{a}|{b}": _greedy(regions[a], regions[b], thresh) for a, b in PAIRS}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for key, mm in m.items():
        a, b = key.split("|")
        for i, (j, sc) in mm.items():
            parent[find((a, i))] = find((b, j))
            edges.append(((a, i), (b, j), key, sc))
    comps = {}
    for c in CODES:
        for i in range(len(regions[c])):
            comps.setdefault(find((c, i)), []).append((c, i))
    sets = []
    for nodes in comps.values():
        present = [c for c in CODES if any(x[0] == c for x in nodes)]
        if len(present) == 1:
            continue  # unmatched region → counted under "only"
        kind = "all3" if len(present) == 3 else "|".join(present)
        ov = {}
        for u, v, key, sc in edges:
            if u in nodes:
                ov[key] = max(ov.get(key, 0.0), sc)
        sets.append({"kind": kind, "nodes": sorted(nodes), "overlaps": ov,
                     "chained": kind == "all3" and len(ov) < 3})
    order = {"all3": 0, "L|O": 1, "L|C": 2, "O|C": 3}
    sets.sort(key=lambda st: (order[st["kind"]], min(regions[c][i]["box"]["y1"] for c, i in st["nodes"])))
    out_sets = []
    for k, st in enumerate(sets, 1):
        members = {}
        for c, i in st["nodes"]:
            members.setdefault(c, []).append(_member(c, i, regions[c][i]))
        out_sets.append({"id": f"S{k}", "kind": st["kind"], "chained": st["chained"],
                         "overlaps": {p: round(v, 3) for p, v in st["overlaps"].items()}, "members": members})
    in_set = {c: sum(len(st["members"].get(c, [])) for st in out_sets) for c in CODES}
    # region-level view for the heatmap: how many of A's regions sit in a set that also has B
    with_b = {f"{a}>{b}": sum(len(st["members"].get(a, [])) for st in out_sets if b in st["members"])
              for a in CODES for b in CODES if a != b}
    counts = {"regions": {c: len(regions[c]) for c in CODES},
              "all3": sum(s_["kind"] == "all3" for s_ in out_sets),
              "chained": sum(s_["chained"] for s_ in out_sets),
              "pairs": {f"{a}|{b}": sum(s_["kind"] == f"{a}|{b}" for s_ in out_sets) for a, b in PAIRS},
              "only": {c: len(regions[c]) - in_set[c] for c in CODES},
              "regions_in": {c: {"all3": sum(len(s_["members"].get(c, [])) for s_ in out_sets if s_["kind"] == "all3"),
                                 **{f"{a}|{b}": sum(len(s_["members"].get(c, [])) for s_ in out_sets
                                                    if s_["kind"] == f"{a}|{b}") for a, b in PAIRS}}
                             for c in CODES},
              "with": with_b}
    return {"overlap_thresh": thresh, "sets": out_sets, "counts": counts}


# ── 3B: material agreement (LLM judge) ────────────────────────────────────────

def _fmt_member(c: str, mb: dict) -> str:
    els = ",".join(mb["elements"])
    if c == "L":
        name = "unnamed (no library match)" if unnamed(mb["material"]) else f'{mb["material"]} (library, {mb["confidence"]})'
        ratios = ", ".join(f"{k} {v}%" for k, v in sorted(mb["ratios"].items(), key=lambda t: -t[1]))
        return f"  L {mb['id']} | elements {els} | name: {name}" + (f" | ratios: {ratios}" if ratios else "")
    alts = "; ".join(a.get("name", "") if isinstance(a, dict) else str(a) for a in mb["alternatives"])
    return (f"  {c} {mb['id']} | elements {els} | name: {mb['material']} ({mb['confidence']}) | "
            f"reasoning: {mb['reasoning']}" + (f" | alternatives: {alts}" if alts else ""))


def build_prompt(sets: list, sample_context: str) -> str:
    lines = []
    if sample_context:
        lines += [f"Sample: {sample_context}.", ""]
    lines += ["Methods:",
              "  L = OpenCV + Library: deterministic match of element ratios against a materials library "
              "(no written reasoning — its evidence is the element list and ratios).",
              "  O = OpenCV + LLM: an LLM named OpenCV's multi-element regions.",
              "  C = Car VLM + LLM: an LLM named the Car VLM's multi-element regions.", "",
              "Each set below is regions from different methods at the SAME location:", ""]
    for s in sets:
        who = "all 3" if s["kind"] == "all3" else " + ".join(s["kind"].split("|"))
        lines.append(f"{s['id']} [{who}]")
        lines += [_fmt_member(c, mb) for c in CODES for mb in s["members"].get(c, [])]
        lines.append("")
    lines += [RULES, "", f"Return JSON with this schema:\n{SCHEMA}"]
    return "\n".join(lines)


def _parse(text: str) -> dict:
    t = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    a, b = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[a:b + 1]) if 0 <= a < b else {}
    except json.JSONDecodeError:
        return {}


def apply_judgement(l3: dict, reply: dict) -> int:
    """Copy the LLM verdicts onto the sets (with sanity rules) and fill 3B counts. Returns # sets judged."""
    verdicts = {str(v.get("set")): v for v in reply.get("sets", []) if isinstance(v, dict)}
    judged = 0
    for s in l3["sets"]:
        v = verdicts.get(s["id"])
        if v is None:
            s.update(match=None, agree=[], note="not judged by the LLM")
            continue
        judged += 1
        match = v.get("match") if v.get("match") in MATCH_TYPES else "none"
        agree = [c for c in CODES if c in (v.get("agree") or []) and c in s["members"]]
        if "L" in agree and all(unnamed(mb["material"]) for mb in s["members"]["L"]):
            agree.remove("L")  # an unnamed library region is a location match only
        if len(agree) < 2:
            match, agree = "none", []
        s.update(match=match, agree=agree, note=str(v.get("note", "")))
    mc = {t: {"all3": 0, "L|O": 0, "L|C": 0, "O|C": 0} for t in MATCH_TYPES}
    for s in l3["sets"]:
        if s.get("match") is None:
            continue
        who = "all3" if len(s["agree"]) == 3 else "|".join(s["agree"]) if s["agree"] else None
        if who:
            mc[s["match"]][who] += 1
        else:
            mc["none"][s["kind"]] += 1
    l3["material_counts"] = mc
    l3["n_judged"] = judged
    return judged


def judge_sample(l3: dict, sample_context: str, model: str, provider: str, key: str, a_url: str, o_url: str,
                 out_dir: pathlib.Path, max_tokens: int = 8192) -> dict:
    """One LLM call for one sample's sets. Writes level3_prompt.txt / level3_reply.txt / level3.json."""
    from automap_hxn import vlm_pipeline
    prompt = build_prompt(l3["sets"], sample_context)
    (out_dir / "level3_prompt.txt").write_text(f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{prompt}")
    meta, t0, err = {}, time.time(), None
    if l3["sets"]:
        try:
            text = vlm_pipeline.query_llm_text(SYSTEM, prompt, model, provider, key or "", anthropic_url=a_url,
                                               openai_url=o_url, max_tokens=max_tokens, meta=meta)
            (out_dir / "level3_reply.txt").write_text(text)
            reply = _parse(text)
            meta["parse_status"] = "ok" if reply.get("sets") else "failed"
        except Exception as ex:  # keep the 3A result even if the judge fails
            err, reply = f"{type(ex).__name__}: {ex}", {}
    else:
        reply = {"sets": []}
    apply_judgement(l3, reply)
    l3["judge"] = {"model": model, "error": err, "prompt_file": "level3_prompt.txt",
                   "reply_file": "level3_reply.txt" if l3["sets"] and not err else None}
    (out_dir / "level3.json").write_text(json.dumps(l3, indent=2))
    return {"meta": meta, "time_s": time.time() - t0, "error": err}


def method_ids(phase_methods: list) -> "dict | None":
    """Pick L / O / C method ids from a run's phase methods, or None if any is missing."""
    lib = next((m for m in phase_methods if m.startswith("opencv:") and m.endswith(":standard")), None)
    ids = {"L": lib, "O": f"{lib}:llm" if lib else None, "C": "vlm:car_standalone"}
    return ids if all(ids[c] in phase_methods for c in CODES) else None


def call_record(meta: dict, dt: float, price, model: str, error, raw_path) -> dict:
    tin, tout = meta.get("input_tokens"), meta.get("output_tokens")
    cost = (tin * price[0] + tout * price[1]) / 1e6 if (price and tin is not None and tout is not None) else None
    return {"kind": "level3", "method": "level3", "element": None, "time_s": round(dt, 2),
            "stop_reason": meta.get("stop_reason"), "truncated": bool(meta.get("truncated")),
            "parse_status": "error" if error else meta.get("parse_status"),
            "input_tokens": tin, "output_tokens": tout, "cost_usd": round(cost, 5) if cost is not None else None,
            "error": error, "raw_reply": raw_path}


def run_level3(results: dict, run_dir: pathlib.Path, cfg: dict, model: str, sample_context: str,
               pricing: dict, workers: int = 10) -> list:
    """Compute Level 3 for every sample in `results` (in place). Returns the API call records."""
    from automap_hxn import vlm_pipeline
    ids = method_ids(results.get("phase_methods", []))
    results["level3_methods"] = ids
    if not ids:
        return []
    thresh = float(results.get("phase_match_overlap", 0.5))
    model = cfg.get("model") or model
    provider = vlm_pipeline.get_provider(model)
    key, a_url, o_url = vlm_pipeline.resolve_key(provider)
    jobs = {}
    for s, sd in results["samples"].items():
        regs = {c: sd["phase"]["regions"].get(ids[c], []) for c in CODES}
        sd["level3"] = location_sets(regs, thresh)
        jobs[s] = sd["level3"]
    calls = []

    def _one(s):
        return s, judge_sample(jobs[s], sample_context, model, provider, key, a_url, o_url, run_dir / s,
                               int(cfg.get("max_tokens", 8192)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for s, r in ex.map(_one, jobs):
            raw = f"{s}/level3_reply.txt" if (run_dir / s / "level3_reply.txt").exists() else None
            calls.append(call_record(r["meta"], r["time_s"], pricing.get(model), model, r["error"], raw))
            l3 = jobs[s]
            print(f"  [{s}] 3A: {l3['counts']['all3']} all-3, {sum(l3['counts']['pairs'].values())} pairs · "
                  f"3B: {l3['n_judged']}/{len(l3['sets'])} judged" + (f" · ERROR {r['error']}" if r["error"] else ""))
    return calls


# ── CLI: apply to an existing run ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "src"))
    sys.path.insert(0, str(here))

    rp = run_dir / "results.json"
    results = json.loads(rp.read_text())
    backup = run_dir / "results.before_level3.json"
    if not backup.exists():
        shutil.copy2(rp, backup)
        print(f"backup → {backup}")
    cfg_all = results.get("config", {})
    cfg = cfg_all.get("level3") or cfg_all.get("reconcile") or {}
    model = results.get("vlm_model") or "claude-sonnet-4-6"
    ctx = cfg_all.get("vlm", {}).get("sample_context", "")
    price = results.get("pricing_per_mtok")
    pricing = {model: price} if isinstance(price, list) else (price or {})
    if not method_ids(results.get("phase_methods", [])):
        sys.exit("[ERROR] run lacks OpenCV + Library, OpenCV + LLM or Car VLM + LLM material regions")

    print(f"Level 3 on {run_dir.name}: {len(results['samples'])} sample(s), model {cfg.get('model') or model}")
    results["calls"] = [c for c in results.get("calls", []) if c.get("kind") != "level3"] + \
        run_level3(results, run_dir, cfg, model, ctx, pricing)
    rp.write_text(json.dumps(results, indent=2))

    from report import write_report, write_runs_index
    write_report(results, run_dir / "index.html")
    write_runs_index(run_dir.parent)
    print(f"Report → {run_dir / 'index.html'}")
    if (run_dir / "car_vs_opencv.html").exists():
        import car_vs_opencv_report
        print("Report →", car_vs_opencv_report.build(run_dir))


if __name__ == "__main__":
    main()
