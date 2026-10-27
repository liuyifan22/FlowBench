#!/usr/bin/env python3
"""Apply the occlusion fix in place to ithor_v11 / procthor_v11 (see README.md).

Rule (``cit_pipeline.color_som.fix_occlusion_row``): drop occlusion questions whose occludee
is < 150 px in the before image; label any mover cover > 0 as "occluded"; reword the choice to
"Partly or fully hidden (occluded).". Applied identically to every file holding an id.

  python apply_v11_occlusion_fix.py --target ithor
  python apply_v11_occlusion_fix.py --target procthor
Afterwards re-run ``reshuffle_v11_answer_letters.py --target <t> --categories occlusion``.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import shutil
from pathlib import Path

from cit_pipeline.color_som import find_question, fix_occlusion_row

D = Path(os.environ.get("CIT_DATA_ROOT", "data/cit"))


def load(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def dump(p: Path, rows: list[dict]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def gt_metas(rows: list[dict], sample_root: Path) -> dict[str, dict]:
    out, cache = {}, {}
    for r in rows:
        if r["category"] != "occlusion":
            continue
        sid = r["sample_id"]
        if sid not in cache:
            cache = {sid: json.loads((sample_root / sid / "sample.json").read_text())}
        _, q = find_question(cache[sid], r)
        out[r["id"]] = q.get("gt_meta") or {}
    return out


def fix_files(files: list[Path], metas: dict[str, dict]) -> tuple[dict[str, dict], set[str], set[str]]:
    """Rewrite occlusion rows in every file; return (dropped rows of files[0], dropped ids, relabelled ids)."""
    dropped_rows, dropped, relabelled = {}, set(), set()
    for i, f in enumerate(files):
        rows, out = load(f), []
        for r in rows:
            if r["category"] == "occlusion":
                fixed = fix_occlusion_row(r, metas[r["id"]])
                if fixed is None:
                    dropped.add(r["id"])
                    if i == 0:
                        dropped_rows[r["id"]] = {**r, "removed_reason": "occlusion_tiny_occludee"}
                    continue
                if fixed["occlusion_relabelled"]:
                    relabelled.add(r["id"])
                r = fixed
            out.append(r)
        dump(f, out)
        print(f"  {f.relative_to(D)}: {len(rows)} -> {len(out)}", flush=True)
    return dropped_rows, dropped, relabelled


def ithor() -> None:
    R = D / "ithor_v11"
    som_rows = load(R / "som/dataset.jsonl")
    metas = gt_metas(som_rows, R / "natural/samples")
    dropped_rows, dropped, relabelled = fix_files([R / "som/dataset.jsonl", R / "natural/dataset.jsonl"], metas)
    nat_pre = {r["id"]: r for r in load(R / "natural/dataset.jsonl")}  # already fixed; only for paths
    for r in dropped_rows.values():
        for k in ("image_before", "image_after"):
            (R / "som" / r[k]).unlink(missing_ok=True)
            # natural image names: same stem without the _som suffix
            (R / "natural" / r[k].replace("_som.png", ".png")).unlink(missing_ok=True)
    removed = load(R / "som/removed.jsonl") + list(dropped_rows.values())
    dump(R / "som/removed.jsonl", removed)
    som = load(R / "som/dataset.jsonl")
    used = {r["sample_id"] for r in som}
    n_dirs = 0
    for d in (R / "natural/samples").iterdir():
        if d.name not in used:
            shutil.rmtree(d)
            n_dirs += 1
    m = json.loads((R / "build_meta.json").read_text())
    m.update({
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_rows": len(som),
        "rows_per_category": dict(sorted(collections.Counter(r["category"] for r in som).items())),
        "rows_per_source_version": dict(collections.Counter(r["source_version"] for r in som)),
        "removed": dict(collections.Counter(r["removed_reason"] for r in removed)),
        "n_samples": len(used),
        "occlusion_fix": {"min_occludee_px": 150, "relabelled_to_occluded": len(relabelled),
                          "dropped": len(dropped)},
    })
    (R / "build_meta.json").write_text(json.dumps(m, indent=2))
    print(f"ithor: dropped {len(dropped)}, relabelled {len(relabelled)}, removed {n_dirs} unused sample dirs, "
          f"{len(som)} rows left", flush=True)
    del nat_pre


def procthor() -> None:
    V = D / "procthor_v11"
    files = [V / "som/dataset.jsonl"] + [V / f"{s}.jsonl" for s in ("test", "val", "train0", "train1", "train2")]
    som_rows = load(files[0])
    metas = gt_metas(som_rows, V / "natural")
    dropped_rows, dropped, relabelled = fix_files(files, metas)
    for r in dropped_rows.values():
        for k in ("image_before", "image_after"):
            (V / "som" / r[k]).unlink(missing_ok=True)
    dump(V / "som/dataset_removed_occlusion.jsonl", list(dropped_rows.values()))
    # Reasoning: dropped rows have no codes; relabelled rows' CoT argues for the old answer.
    allow = V / "filters/som_cot_ids.txt"
    ids = [l.strip() for l in open(allow) if l.strip()]
    keep = [i for i in ids if i not in dropped and i not in relabelled]
    allow.write_text("\n".join(keep) + "\n")
    (V / "filters/occlusion_fix_ids.json").write_text(json.dumps(
        {"dropped": sorted(dropped), "relabelled": sorted(relabelled)}))
    som = load(files[0])
    bm = json.loads((V / "som/build_meta.json").read_text())
    bm.update({"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "n_rows": len(som),
               "rows_per_category": dict(sorted(collections.Counter(
                   f"{r['source_version']}/{r['category']}" for r in som).items())),
               "occlusion_fix": {"min_occludee_px": 150, "dropped": len(dropped),
                                 "relabelled_to_occluded": len(relabelled)}})
    (V / "som/build_meta.json").write_text(json.dumps(bm, indent=2))
    sm = json.loads((V / "cit_psi2_split_meta.json").read_text())
    sm["splits"] = {s: sum(1 for _ in open(V / f"{s}.jsonl")) for s in sm["splits"]}
    sm["n_usable"] = len(som)
    sm["filter"] += " + occlusion fix (occludee >= 150 px, any cover = occluded)"
    (V / "cit_psi2_split_meta.json").write_text(json.dumps(sm, indent=2))
    print(f"procthor: dropped {len(dropped)}, relabelled {len(relabelled)}, allowlist {len(ids)} -> {len(keep)}, "
          f"{len(som)} rows left; splits {sm['splits']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("ithor", "procthor"), required=True)
    {"ithor": ithor, "procthor": procthor}[ap.parse_args().target]()
