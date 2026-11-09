"""
LLM material reconciliation — one pass over every method's material regions for a sample.

The LLM groups regions from different methods that describe the same particle / phase
(same place + compatible chemistry, even when named differently or not named at all).
Which methods found / missed each group is then computed here from the region IDs,
so the bookkeeping does not depend on the LLM getting counts right.
"""

import json
import string

SYSTEM = (
    "You are an XRF (X-ray fluorescence) materials scientist reconciling material / phase detections "
    "that several analysis methods produced on the SAME sample scan. Some methods are deterministic "
    "(blob detection + a materials library), others are vision-language models with an LLM naming step. "
    "Your job is to decide which regions from different methods refer to the same physical particle or "
    "phase, so agreement between methods can be measured. Be concise. Return ONLY valid JSON."
)

RULES = """RULES
1. Group regions that refer to the same particle/phase: boxes overlap or are immediately adjacent AND the
   element sets are compatible (one may list a subset of the other's elements).
2. Names differ between methods: treat chemically equivalent names as the same material (e.g. a mineral name
   vs its formula, "Fe oxide" vs "hematite"). If names are related but not equivalent (e.g. gypsum vs anhydrite),
   still group them when location + elements match, and say so in the note.
3. Regions marked "unnamed" were detected but not identified — still group them by location + elements.
4. Every region ID must appear in exactly one group. A region nobody else found is its own group.
5. "material": your best canonical name for the group (use the most specific name the evidence supports;
   "unidentified <elements> phase" if no method named it).
6. "note": ONE short sentence — why grouped, and any naming disagreement. Omit if trivial.
7. "overall": 2–3 sentences on where deterministic methods and VLM methods agree or disagree."""

SCHEMA = ('{"groups": [{"material": "gypsum (CaSO4·2H2O)", "regions": ["A3", "C1", "V2.4"], '
          '"note": "A3 unnamed; V2.4 calls it calcium sulfate."}], "overall": "..."}')


def _codes(methods: list) -> dict:
    """Short region-ID prefixes: AutoMap methods A, B, C…; VLM variants V1, V2…"""
    out, letters, v = {}, iter(string.ascii_uppercase), 0
    for m in methods:
        if m.startswith("vlm:"):
            v += 1
            out[m] = f"V{v}."
        else:
            out[m] = next(letters)
    return out


def _desc(m: str, model_input: dict) -> str:
    kind = "VLM + LLM naming" if m.startswith("vlm:") else "deterministic: blob unions + materials library"
    return f"{kind}; input: {model_input.get(m, '?')}"


def build_prompt(regions_by_method: dict, model_input: dict, sample_context: str) -> "tuple[str, dict]":
    """Returns (prompt, id_map) where id_map[region_id] = (method, index into that method's region list)."""
    methods = [m for m in regions_by_method]
    codes = _codes(methods)
    id_map, lines = {}, []
    if sample_context:
        lines += [f"Sample: {sample_context}.", ""]
    lines += ["Box coordinates are fractions of the image (x1,y1,x2,y2), (0,0) = top-left.", ""]
    for m in methods:
        regs = regions_by_method[m]
        lines.append(f"METHOD {codes[m].rstrip('.')} = {m}  ({_desc(m, model_input)})  — {len(regs)} region(s)")
        for k, r in enumerate(regs, start=1):
            rid = f"{codes[m]}{k}"
            id_map[rid] = (m, k - 1)
            b = r["box"]
            name = str(r.get("material") or "")
            if not name or name.lower().startswith("no library match"):
                name = "unnamed"
            lines.append(f"  {rid} | {','.join(r['elements'])} | {b['x1']:.2f},{b['y1']:.2f},"
                         f"{b['x2']:.2f},{b['y2']:.2f} | {name} ({r.get('confidence', '—')})")
        lines.append("")
    lines += [RULES, "", f"Return JSON with this schema:\n{SCHEMA}"]
    return "\n".join(lines), id_map


def parse_reply(text: str) -> dict:
    t = text
    if "```" in t:
        t = "\n".join(l for l in t.splitlines() if not l.strip().startswith("```"))
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return {}
    try:
        return json.loads(t[a:b + 1])
    except json.JSONDecodeError:
        return {}


def postprocess(reply: dict, id_map: dict, regions_by_method: dict) -> dict:
    """Attach method/name details to each group, compute found-by / missing-from and agreement class."""
    methods = list(regions_by_method)
    det = [m for m in methods if not m.startswith("vlm:")]
    vlm = [m for m in methods if m.startswith("vlm:")]
    seen, groups = set(), []

    def _make(material, rids, note, source="llm"):
        members = []
        for rid in rids:
            if rid not in id_map or rid in seen:
                continue
            seen.add(rid)
            m, i = id_map[rid]
            r = regions_by_method[m][i]
            members.append({"region_id": rid, "method": m, "elements": r["elements"],
                            "name_used": r.get("material") or "", "confidence": r.get("confidence"),
                            "box": r["box"]})
        if not members:
            return
        found = sorted({x["method"] for x in members}, key=methods.index)
        fd, fv = [m for m in found if m in det], [m for m in found if m in vlm]
        if len(found) == len(methods):
            cls = "all"
        elif len(found) == 1:
            cls = "single"
        elif fd and fv:
            cls = "det+vlm"
        elif fd:
            cls = "det-only"
        else:
            cls = "vlm-only"
        groups.append({"id": len(groups) + 1, "material": material, "note": note, "source": source,
                       "elements": sorted({e for x in members for e in x["elements"]}),
                       "found_by": found, "missing_from": [m for m in methods if m not in found],
                       "agreement": cls, "members": members})

    for g in reply.get("groups", []):
        _make(str(g.get("material", "")), [str(r) for r in g.get("regions", [])], str(g.get("note", "")))
    # anything the LLM left out becomes its own group, so nothing silently disappears
    for rid in id_map:
        if rid not in seen:
            m, i = id_map[rid]
            r = regions_by_method[m][i]
            _make(r.get("material") or "unnamed", [rid], "not assigned by the LLM", source="unassigned")

    # per-method and pairwise summaries
    per_method = {}
    for m in methods:
        mine = [g for g in groups if m in g["found_by"]]
        per_method[m] = {
            "regions": len(regions_by_method[m]), "groups": len(mine),
            "shared": sum(len(g["found_by"]) > 1 for g in mine),
            "unique": sum(len(g["found_by"]) == 1 for g in mine),
            "shared_with_det": sum(any(x in det and x != m for x in g["found_by"]) for g in mine),
            "shared_with_vlm": sum(any(x in vlm and x != m for x in g["found_by"]) for g in mine),
        }
    pairwise = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            ga = {g["id"] for g in groups if a in g["found_by"]}
            gb = {g["id"] for g in groups if b in g["found_by"]}
            u = ga | gb
            pairwise[f"{a}|{b}"] = {"both": len(ga & gb), "jaccard": (len(ga & gb) / len(u)) if u else 1.0}
    return {"overall": str(reply.get("overall", "")), "groups": groups,
            "per_method": per_method, "pairwise": pairwise,
            "n_unassigned": sum(g["source"] == "unassigned" for g in groups)}
