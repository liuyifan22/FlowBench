#!/usr/bin/env python3
"""Recover c_gt (rule-based reasoning from simulator metadata) for every FlowBench-Sim question and style.

The stored "cold" CoTs (cit_pipeline/cot.py) were broken for five categories:
  count_near, near_object   no template at all -> one-line stub "Question type .. Move .. Answer .."
  left_right_landmark       template reads bearing_* fields the generator never wrote -> "None° > None°"
  feasibility, contact_wall "Probe orthogonal ... = None" / "successful wall steps = None"
  closer_to                 "an absent type (None)" when all four options are real; also calls the
                            bounding-box surface distance a "3D centroid distance"
  occlusion                 states the old 0.30 cover threshold, but v11 counts any cover as occluded
This script rebuilds c_gt from sample.json gt_meta with the rule each category actually used, in the
row's own wording: natural rows name objects as the natural question does; colour_som rows add
"(marked <colour>)"; red_som rows add the badge label "(1)", "(A)". The option set is the row's own
(closer_to SoM rows can offer a different fourth object than the natural row). Every c_gt's conclusion
is checked against the row's answer; rows that fail the check get no c_gt and are counted.

Rows whose recorded metadata cannot support their answer get no c_gt. In 1,814 closest_three rows the stored answer
contradicts the rendered after image (the stored distances agree with the image), and in 53 natural rows two
options have the same text.

Only metadata that was recorded is used. count_near stores counts, not per-object distances, so its
c_gt gives the counts; near_object gives the sets of objects within the threshold.

  python recover_cgt.py            -> outputs/cgt/cgt_<split>.jsonl  {id, natural, colour_som, red_som}
"""
from __future__ import annotations

import collections, json, os, re, sys
from multiprocessing import Pool
from pathlib import Path

SIM = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release") + "/sim")
OUT = Path("outputs/cgt")
STYLES = ("natural", "colour_som", "red_som")


def f2(x):
    return f"{x:.2f}"


def type_name(oid: str) -> str:
    t = oid.split("|")[0]
    return "the " + re.sub(r"(?<!^)(?=[A-Z])", "", t).lower()


def strip_refs(s: str) -> str:
    return re.sub(r"\s*\((?:marked \w+|\d|[A-F])\)", "", s).strip()


class Names:
    """object id -> phrase in one style's wording."""

    def __init__(self, style, gm, rows):
        self.style, self.gm = style, gm
        marks = {m["id"]: m["object_id"] for m in (gm.get("som_marks") or [])}
        self.mark_of = {oid: mid for mid, oid in marks.items()}
        col = rows["colour_som"]
        self.colour = (col.get("som_colors") or {}) if col else {}
        txt = " ".join(str(col.get(k) or "") for k in ("motion_command", "question", *(f"choice_{L}" for L in "ABCDEF"))) if col else ""
        self.cname = {}
        for m in re.finditer(r"\(marked (\w+)\)", txt):        # the phrase right before each mark
            pre = txt[:m.start()].rstrip()
            k = pre.rfind("the ")
            if k >= 0 and m.group(1) not in self.cname:
                self.cname[m.group(1)] = pre[k:]
        red = rows["red_som"]
        self.label = (red.get("som_labels") or {}) if red else {}
        self.nat = {}
        for a, b in (("landmark_a_id", "phrase_a"), ("landmark_b_id", "phrase_b"), ("far_id", "far_phrase"),
                     ("fourth_id", "fourth_phrase"), ("landmark_id", "landmark_phrase"), ("occludee_id", "occludee_phrase")):
            if gm.get(a) and gm.get(b):
                self.nat[gm[a]] = gm[b]
        for oid, ph in zip(gm.get("candidate_ids") or [], gm.get("candidate_pool") or []):
            self.nat[oid] = ph

    def __call__(self, oid, mover_phrase=None):
        if self.style == "natural":
            return mover_phrase or self.nat.get(oid) or type_name(oid)
        mid = self.mark_of.get(oid)
        base = None
        if mid and self.colour.get(mid):
            base = self.cname.get(self.colour[mid])
        base = base or strip_refs(mover_phrase or self.nat.get(oid) or type_name(oid))
        if mid is None:
            return base
        if self.style == "colour_som" and self.colour.get(mid):
            return f"{base} (marked {self.colour[mid]})"
        if self.style == "red_som" and self.label.get(mid):
            return f"{base} ({self.label[mid]})"
        return base


def ans_text(row):
    letters = row["answer"]
    opts = [row.get(f"choice_{L}") or "" for L in letters]
    return ", ".join(o.rstrip(".") for o in opts)


def option_ids(row, style, gm, names):
    """[(letter, option text, object id or None)] for object-valued options."""
    out = []
    for L in "ABCDEF":
        t = row.get(f"choice_{L}")
        if not t:
            continue
        oid = None
        if style == "natural":
            core = t.strip().rstrip(".")
            for k, v in names.nat.items():
                if v == core:
                    oid = k
        elif style == "colour_som":
            m = re.search(r"\(marked (\w+)\)", t)
            if m:
                mid = next((k for k, c in names.colour.items() if c == m.group(1)), None)
                oid = next((o for o, k in names.mark_of.items() if k == mid), None)
        else:
            mid = next((k for k, lab in names.label.items() if lab == L), None)
            oid = next((o for o, k in names.mark_of.items() if k == mid), None)
        out.append((L, t, oid))
    return out


def build(cat, style, row, v, gm, rows):
    mv = v["move"]
    nm = Names(style, gm, rows)
    mover = nm(mv["object_id"], gm.get("mover_phrase") if style == "natural" else None)
    motion = row["motion_command"]
    ans = ans_text(row)
    yes = lambda b: "yes" if b else "no"
    if cat == "closer_to":
        dist = {}
        for a, d in (("landmark_a_id", "distance_a"), ("landmark_b_id", "distance_b"), ("far_id", "far_distance"),
                     ("fourth_id", "fourth_distance")):
            if gm.get(a) is not None and gm.get(d) is not None:
                dist[gm[a]] = float(gm[d])
        opts = option_ids(row, style, gm, nm)
        parts, best = [], None
        for L, t, oid in opts:
            name = nm(oid) if oid else t.rstrip(".")
            if oid in dist:
                parts.append(f"{name} {f2(dist[oid])} m")
                if best is None or dist[oid] < best[1]:
                    best = (L, dist[oid], name)
            else:
                parts.append(f"{name} (not in the room)")
        if best is None or best[0] != row["answer"]:
            return None
        before, after_ = gm.get("closer_before_id"), gm.get("closer_after_id")
        txt = [f"Question type: closer_to. After the move ({motion}), which listed object is {mover} closer to?",
               f"Rule: compare the bounding-box surface distances from {mover} to each listed object after the move; the nearest one is the answer.",
               "Distances after the move: " + "; ".join(parts) + "."]
        if before and after_ and before != after_:
            txt.append(f"Before the move {mover} was closer to {nm(before)} than to {nm(after_)}; the move reverses that.")
        txt += [f"The nearest is {best[2]}.", f"Answer: {ans}"]
        return "\n".join(txt)
    if cat == "closest_three":
        ids, ds = gm.get("candidate_ids") or [], gm.get("candidate_distances") or []
        if len(ids) != len(ds) or not ids:
            return None
        ranked = sorted(zip(ids, ds), key=lambda x: x[1])
        top = {i for i, _ in ranked[:3]}
        opts = option_ids(row, style, gm, nm)
        want = {oid for L, t, oid in opts if L in row["answer"]}
        if top != want:
            return None
        b4 = gm.get("answer_before") or []
        txt = [f"Question type: closest_three. After the move ({motion}), which three listed objects are closest to {mover}?",
               f"Rule: rank the six listed objects by the distance between their 3D centres and the centre of {mover} after the move; take the three nearest.",
               "Distances after the move: " + "; ".join(f"{nm(i)} {f2(d)} m" for i, d in ranked) + "."]
        if b4 and set(b4) != top:
            txt.append("Before the move the three nearest were " + ", ".join(nm(i) for i in b4) + "; the move changes the set.")
        txt += ["The three nearest are " + ", ".join(nm(i) for i, _ in ranked[:3]) + ".", f"Answer: {ans}"]
        return "\n".join(txt)
    if cat == "count_near":
        n0, n1, T = gm.get("count_before"), gm.get("count_after"), gm.get("thresh_m")
        if n1 is None or T is None or str(n1) != ans.strip():
            return None
        objs = ", ".join(nm(i) for i in gm.get("candidate_ids") or [])
        return "\n".join([
            f"Question type: count_near. After the move ({motion}), how many of the listed objects are within {f2(T)}m of {mover}?",
            f"Rule: count the listed objects ({objs}) whose 3D position is within {f2(T)} m of the 3D position of {mover}.",
            f"Before the move {n0} of the {len(gm.get('candidate_ids') or [])} objects are within {f2(T)} m; after the move {n1} are.",
            f"Answer: {ans}"])
    if cat == "near_object":
        T, lid = gm.get("thresh_m"), gm.get("landmark_id")
        s0, s1 = gm.get("set0") or [], gm.get("set1") or []
        if T is None or lid is None:
            return None
        inside = lid in s1
        if inside != bool(gm.get("answer_after", v["questions"][0].get("answer"))):
            return None
        def fmt(s):
            if not s:
                return "none"
            names = [nm(i) for i in s]
            cnt = collections.Counter(names)
            out, seen = [], set()
            for n in names:
                if n in seen:
                    continue
                seen.add(n)
                out.append(n if cnt[n] == 1 else f"{cnt[n]} {n.removeprefix('the ')}s")
            return ", ".join(out)
        return "\n".join([
            f"Question type: near_object. After the move ({motion}), is {mover} within {f2(T)}m of {nm(lid)}?",
            f"Rule: yes if the distance between the 3D positions of {mover} and {nm(lid)} is at most {f2(T)} m.",
            f"Objects within {f2(T)} m of {mover} before the move: {fmt(s0)}; after the move: {fmt(s1)}.",
            f"{nm(lid)[0].upper() + nm(lid)[1:]} is {'' if inside else 'not '}within {f2(T)} m after the move, so the answer is {yes(inside)}.",
            f"Answer: {ans}"])
    if cat == "left_right_landmark":
        u0, u1, uL = gm.get("moved_screen_u_before"), gm.get("moved_screen_u_after"), gm.get("landmark_screen_u")
        asked, lid = gm.get("asked"), gm.get("landmark_id")
        if None in (u1, uL, asked, lid):
            return None
        truth = (u1 < uL) if asked == "left" else (u1 > uL)
        if truth != bool(gm.get("answer_after", v["questions"][0].get("answer"))):
            return None
        side = "left" if u1 < uL else "right"
        W = 800
        return "\n".join([
            f"Question type: left_right_landmark. After the move ({motion}), is {mover} to the {asked} of {nm(lid)} from the camera's viewpoint?",
            f"Rule: compare the horizontal image position (x of the object's mask centre, 0 = left edge, {W} = right edge) in the after image; smaller x is farther left.",
            (f"{mover[0].upper() + mover[1:]}: x = {u0:.0f} px before the move, {u1:.0f} px after; " if u0 is not None else
             f"{mover[0].upper() + mover[1:]}: x = {u1:.0f} px after the move; ") + f"{nm(lid)}: x = {uL:.0f} px.",
            f"After the move {mover} is to the {side} of {nm(lid)}, so the answer is {yes(truth)}.",
            f"Answer: {ans}"])
    if cat == "occlusion":
        cover, px = row.get("occlusion_cover"), row.get("occludee_px")
        if cover is None:
            cover = gm.get("occlusion_rate")
        if px is None:
            px = gm.get("occludee_mask_pixels_rgb0")
        oid = gm.get("occludee_id")
        if cover is None or oid is None:
            return None
        occluded = cover > 0
        hidden_opt = next((row[f"choice_{L}"] for L in "AB" if "hidden" in (row.get(f"choice_{L}") or "").lower()), None)
        if hidden_opt is None or (row[f"choice_{row['answer']}"] == hidden_opt) != occluded:
            return None
        return "\n".join([
            f"Question type: occlusion. After the move ({motion}), is {nm(oid)} partly or fully hidden by {mover}, or totally visible?",
            f"Rule: measure the fraction of {nm(oid)}'s pixels in the before image that {mover} covers in the after image (instance segmentation); any cover means partly or fully hidden.",
            f"{nm(oid)[0].upper() + nm(oid)[1:]} has {int(px)} pixels in the before image; after the move {mover} covers {100 * cover:.1f}% of them.",
            f"So {nm(oid)} is {'partly or fully hidden' if occluded else 'totally visible'}.",
            f"Answer: {ans}"])
    if cat == "feasibility":
        ok, pm, pd = gm.get("teleport_success"), gm.get("probe_meters"), gm.get("probe_direction_phrase") or gm.get("probe_direction")
        if ok is None or pm is None or bool(ok) != bool(gm.get("answer_after", v["questions"][0].get("answer"))):
            return None
        return "\n".join([
            f"Question type: feasibility. Starting from the after state ({motion}), can {mover} move a further {f2(pm)} m {pd} without colliding?",
            f"Rule: from the after pose, attempt a collision-checked move of {f2(pm)} m {pd} in the simulator (no forced placement), then restore; feasible if the simulator accepts it.",
            f"The simulator {'accepts' if ok else 'rejects'} the move (lastActionSuccess = {bool(ok)}), so it is {'feasible' if ok else 'not feasible'}.",
            f"Answer: {ans}"])
    if cat == "contact_wall":
        c0, c1, thr = gm.get("min_edge_clearance_before_m"), gm.get("min_edge_clearance_m"), gm.get("threshold_m")
        blocked = bool(gm.get("wall_blocked_next_step"))
        if c1 is None or thr is None:
            return None
        truth = blocked or c1 < thr
        if truth != bool(gm.get("answer_after", v["questions"][0].get("answer"))):
            return None
        return "\n".join([
            f"Question type: contact_wall. After the move ({motion}), is {mover} in contact with a wall?",
            f"Rule: in contact if the next small step toward the wall is blocked, or if the smallest gap between {mover}'s bounding box and a wall is below {f2(thr)} m.",
            (f"Gap to the nearest wall: {f2(c0)} m before the move, {f2(c1)} m after" if c0 is not None else
             f"Gap to the nearest wall after the move: {f2(c1)} m") + f"; next step blocked: {yes(blocked)}.",
            f"So {mover} is {'' if truth else 'not '}in contact with a wall.",
            f"Answer: {ans}"])
    return None


def work(args):
    split, sid, items = args
    s = json.load(open(SIM / split / "samples" / sid / "sample.json"))
    views = {v["view_id"]: v for v in s["views"]}
    out = []
    for rid, rows in items:
        any_row = next(r for r in rows.values() if r)
        v = views.get(any_row["view_id"])
        rec = {"id": rid, "category": any_row["category"]}
        if v is None:
            rec["error"] = "no_view"; out.append(rec); continue
        gm = v["questions"][0].get("gt_meta") or {}
        for st in STYLES:
            if rows.get(st) is None:
                continue
            try:
                rec[st] = build(any_row["category"], st, rows[st], v, gm, rows)
            except Exception as e:  # noqa: BLE001
                rec[st] = None; rec[f"err_{st}"] = repr(e)[:120]
        out.append(rec)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        by_id = collections.defaultdict(dict)
        for st in STYLES:
            for line in open(SIM / split / st / "dataset.jsonl"):
                r = json.loads(line)
                by_id[r["id"]][st] = r
        by_sample = collections.defaultdict(list)
        for rid, rows in by_id.items():
            for st in STYLES:
                rows.setdefault(st, None)
            by_sample[next(r for r in rows.values() if r)["sample_id"]].append((rid, rows))
        jobs = [(split, sid, items) for sid, items in by_sample.items()]
        stats = collections.defaultdict(collections.Counter)
        with Pool(32) as pool, open(OUT / f"cgt_{split}.jsonl", "w") as fo:
            for recs in pool.imap_unordered(work, jobs, chunksize=64):
                for rec in recs:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    for st in STYLES:
                        if st in rec:
                            stats[rec["category"]][f"{st}_{'ok' if rec[st] else 'fail'}"] += 1
        print(split)
        for c, v in sorted(stats.items()):
            print(f"  {c:22s} " + "  ".join(f"{k} {n}" for k, n in sorted(v.items())))


if __name__ == "__main__":
    main()
