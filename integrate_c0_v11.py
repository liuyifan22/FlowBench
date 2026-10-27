#!/usr/bin/env python3
"""Put the three CoT annotations into procthor_v11 (FlowBench-Sim-Train) after gen_c0_procthor_v11.py.

  procthor_v11/cot/<id>_cold.txt          c_gt  rule-based, from simulator metadata (copied from v9/v10 natural/cot)
  procthor_v11/cot/<id>_warm.txt          c_1   VLM (Qwen3-VL-32B / Qwen3.5-9B) from I_0, I_1 and c_gt (already there)
  procthor_v11/cot/<id>_pretended.txt     c_0   Qwen3.5-9B from I_0, t_m, t_q and c_1 (gen_c0_procthor_v11.py)
  procthor_v11/filters/c0_ids.txt         ids with an accepted c_0 (re-checked here)
  procthor_v11/filters/som_cot_compatible_ids.txt
        ids whose CoTs also fit the colour-SoM row: the CoTs were written for the natural question, and for
        closer_to / closest_three (choices_match_som false) and count_near (different object list) the SoM
        question names other objects
  procthor_v11/filters/cot_meta.json      counts and definitions
  procthor_v11/qwen_training/som_balanced_splits_1B_c0/   1B splits with target = c_0 (+ <answer>X</answer>),
        restricted to c0_ids AND som_cot_compatible_ids

    python integrate_c0_v11.py
"""
from __future__ import annotations

import collections, datetime, importlib.util, json, os, re, shutil
from pathlib import Path

V = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11")
SRC = Path(os.environ.get("CIT_DATA_ROOT", "data/cit"))
REL = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release") + "/sim/train")
COT = V / "cot"
_spec = importlib.util.spec_from_file_location("g", Path(__file__).with_name("gen_c0_procthor_v11.py"))
g = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(g)
_ANSWER_TAIL = re.compile(r"\n\s*Answer\s*:.*\Z", re.IGNORECASE | re.DOTALL)


def som_compatible(nat: dict, som: dict) -> bool:
    """Do the natural and SoM questions ask about the same objects?
    closer_to / closest_three: the options must be the same objects (choices_match_som); closer_to SoM rows
    often replace the natural row's absent distractor with a real object. count_near: both rows come from the
    same sample.json question, so the listed objects are the same ids (checked: 332/332); only the wording
    differs ("the 1st kettle from the left" vs "the kettle (marked brown)"). The 2026-10-19 version compared
    the wordings and wrongly dropped almost every count_near row."""
    cat = som["category"]
    if cat in ("closer_to", "closest_three"):
        return bool(nat.get("choices_match_som"))
    return True


def target(c0: str, letter: str) -> str:
    text = _ANSWER_TAIL.sub("", c0.strip()).strip()
    m = re.search(r"\bAnswer\s*:", text, flags=re.IGNORECASE)
    if m:
        text = text[: m.start()].rstrip()
    return f"{text}\n\n<answer>{letter}</answer>"


def main() -> None:
    allow = open(V / "filters/som_cot_ids.txt").read().split()
    nat = {}
    for line in open(REL / "natural/dataset.jsonl"):
        r = json.loads(line)
        nat[r["id"]] = r
    som = {}
    for line in open(V / "som/dataset.jsonl"):
        r = json.loads(line)
        som[r["id"]] = r
    # 1) c_0: accepted by the generator and re-checked here
    status = {}
    for line in open(COT / "_c0_log.jsonl"):
        rec = json.loads(line)
        status[rec["id"]] = rec["status"]
    ok, recheck_fail = [], collections.Counter()
    for rid in allow:
        p = COT / f"{rid}_pretended.txt"
        if status.get(rid) != "ok" or not p.exists() or rid not in nat:
            continue
        warm = (COT / f"{rid}_warm.txt").read_text().replace("<|endoftext|>", "").strip()
        n = nat[rid]
        choices = {L: n[f"choice_{L}"] for L in "ABCDEF" if n.get(f"choice_{L}")}
        probs = g.check(p.read_text(), g.last_answer(warm), n["category"], n["motion_command"], warm, choices,
                        n["question"])
        if probs:
            recheck_fail[probs[0].split(":")[0]] += 1
            continue
        ok.append(rid)
    # 2) c_gt next to the others
    n_cold = 0
    for rid in allow:
        dst = COT / f"{rid}_cold.txt"
        if dst.exists():
            n_cold += 1
            continue
        ver = som.get(rid, {}).get("source_version") or nat[rid]["source_version"]
        src = SRC / ver / "natural/cot" / f"{rid}_cold.txt"
        if src.exists():
            shutil.copyfile(src, dst); n_cold += 1
    # 3) filters
    compat = [rid for rid in allow if rid in nat and rid in som and som_compatible(nat[rid], som[rid])]
    (V / "filters/c0_ids.txt").write_text("\n".join(ok) + "\n")
    (V / "filters/som_cot_compatible_ids.txt").write_text("\n".join(compat) + "\n")
    okset, cset = set(ok), set(compat)
    by_cat = collections.defaultdict(collections.Counter)
    for rid in allow:
        cat = nat[rid]["category"] if rid in nat else "?"
        by_cat[cat]["allowlist"] += 1
        by_cat[cat]["c0_ok"] += rid in okset
        by_cat[cat]["som_compatible"] += rid in cset
        by_cat[cat]["c0_ok_and_som_compatible"] += (rid in okset and rid in cset)
    meta = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                files={"<id>_cold.txt": "c_gt, rule-based from simulator metadata",
                       "<id>_warm.txt": "c_1, VLM from I_0 + I_1 + c_gt (written for the natural question)",
                       "<id>_pretended.txt": "c_0, Qwen3.5-9B from I_0 + t_m + t_q, c_1 as a private hint"},
                c0_generator="gen_c0_procthor_v11.py", c0_model="Qwen3.5-9B (vLLM 0.27.1, thinking off)",
                n_allowlist=len(allow), n_cold=n_cold, n_c0_ok=len(ok), n_c0_recheck_failed=dict(recheck_fail),
                n_som_compatible=len(compat), n_c0_and_som_compatible=len(okset & cset),
                per_category={k: dict(v) for k, v in sorted(by_cat.items())})
    (V / "filters/cot_meta.json").write_text(json.dumps(meta, indent=2))
    # 4) 1B splits with the c_0 target
    src_dir, out = V / "qwen_training/som_balanced_splits", V / "qwen_training/som_balanced_splits_1B_c0"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "images").exists():
        (out / "images").symlink_to(V / "som/images")
    sm = {}
    for split in ("train", "val", "test", "eval"):
        n_in = n_out = 0
        cats = collections.Counter()
        with open(src_dir / f"{split}.jsonl") as fi, open(out / f"{split}.jsonl", "w") as fo:
            for line in fi:
                n_in += 1
                r = json.loads(line)
                if r["id"] not in okset or r["id"] not in cset:
                    continue
                r["cot_target"] = target((COT / f"{r['id']}_pretended.txt").read_text(), r["answer"])
                r["cot_target_source"] = "c_0 (pretended)"
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1
                cats[r["category"]] += 1
        sm[split] = {"n_in": n_in, "n_out": n_out, "per_category": dict(sorted(cats.items()))}
    (out / "split_meta.json").write_text(json.dumps({
        "source": str(src_dir), "target": "c_0 (pretended CoT, Answer tail removed) + <answer>X</answer>",
        "filters": ["filters/c0_ids.txt", "filters/som_cot_compatible_ids.txt"], "splits": sm}, indent=2))
    print(json.dumps({k: meta[k] for k in ("n_allowlist", "n_cold", "n_c0_ok", "n_c0_recheck_failed",
                                           "n_som_compatible", "n_c0_and_som_compatible")}, indent=1))
    print({k: v["n_out"] for k, v in sm.items()})


if __name__ == "__main__":
    main()
