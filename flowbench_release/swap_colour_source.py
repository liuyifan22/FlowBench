#!/usr/bin/env python3
"""Move the ProcTHOR v11 colour-SoM images from $CIT_DATA_ROOT into the release (mv across filesystems, safely).

1. mirror: <release>/sim/train/_build/procthor_v11_som_images/ gets a hardlink of every release colour image
   plus a copy of the few source images not in the release (dropped questions), i.e. the full source folder.
2. verify: every file in the source folder exists in the mirror with the same size; a random sample is
   byte-compared.
3. swap (--swap): rename the source folder, put a symlink to the mirror in its place, check reads through it,
   then delete the renamed original.
"""
import argparse, filecmp, os, random, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path

SRC = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11/som/images")
REL = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release") + "/sim/train/colour_som/images")
MIRROR = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release") + "/sim/train/_build/procthor_v11_som_images")


def place(names):
    st = {"linked": 0, "copied": 0, "exists": 0}
    for n in names:
        dst = MIRROR / n
        if dst.exists():
            st["exists"] += 1
        elif (REL / n).exists():
            os.link(REL / n, dst); st["linked"] += 1
        else:
            shutil.copyfile(SRC / n, dst); st["copied"] += 1
    return st


def check(names):
    bad = []
    for n in names:
        a, b = SRC / n, MIRROR / n
        if not b.exists() or a.stat().st_size != b.stat().st_size:
            bad.append(n)
    return bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--swap", action="store_true"); ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if SRC.is_symlink():
        print("source is already a symlink ->", os.readlink(SRC)); return
    MIRROR.mkdir(parents=True, exist_ok=True)
    names = sorted(os.listdir(SRC)); print(len(names), "source files", flush=True)
    chunks = [names[i:i + 2000] for i in range(0, len(names), 2000)]
    with Pool(a.workers) as p:
        tot = {}
        for st in p.imap_unordered(place, chunks):
            for k, v in st.items(): tot[k] = tot.get(k, 0) + v
        print("mirror", tot, flush=True)
        bad = [n for b in p.imap_unordered(check, chunks) for n in b]
    print("size mismatches / missing:", len(bad), bad[:5], flush=True)
    sample = random.Random(0).sample(names, 3000)
    diff = [n for n in sample if not filecmp.cmp(SRC / n, MIRROR / n, shallow=False)]
    print("byte-compare sample 3000: differing", len(diff), flush=True)
    if bad or diff:
        sys.exit("verification failed; not swapping")
    if not a.swap:
        print("verified; re-run with --swap to replace the source folder"); return
    old = SRC.with_name("images.__moved_to_flowbench_release")
    os.rename(SRC, old)
    os.symlink(MIRROR, SRC)
    probe = random.Random(1).sample(names, 200)
    assert all((SRC / n).stat().st_size == (old / n).stat().st_size for n in probe), "symlink read check failed"
    print("symlink in place:", SRC, "->", MIRROR, flush=True)
    t0 = time.time(); shutil.rmtree(old)
    print(f"deleted original ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
