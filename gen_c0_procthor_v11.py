#!/usr/bin/env python3
"""Generate c_0 (pretended CoT) for FlowBench-Sim-Train = procthor_v11, with Qwen3.5-9B on vLLM.

c_0 = VLM(t_q, I_0, t_m, c_1): the VLM sees ONLY the before image I_0, the motion t_m and the question
t_q, plus c_1 (the warm CoT, written from I_0 + I_1) as a private hint and the GT answer, and must
reason as if it imagined the outcome. The system prompt and the user prompt are copied verbatim
from ``cit_pipeline.cot`` (PRETENDED_SYSTEM / polish_pretended_cot); the pipeline's own client is
not used only because Qwen3.5 must be called with thinking disabled.

Work set: the reasoning allowlist ``procthor_v11/filters/som_cot_ids.txt`` (c_1 exists and agrees
with the current answer; relabelled occlusion rows are already excluded). Text, image and options
are those of the natural release row (the question c_1 was written for). Output, resumable:

  procthor_v11/cot/<id>_pretended.txt        accepted c_0 (ends with "Answer: <same as c_1>")
  procthor_v11/cot/_c0_log.jsonl             one line per finished id: status, attempts, checks

Checks on every reply (else retried at a higher temperature, up to --max-attempts):
  final "Answer:" equal to c_1's answer (order-free for closest_three); no reference to an
  after / second image, a reference CoT or hints; 25-260 words; no direction contradicting the
  camera-axis motion ("backward" = toward the camera); word-overlap with c_1 <= 0.80.

    python gen_c0_procthor_v11.py --servers-file servers.txt --concurrency 48
    # servers.txt: one http://<host>:<port>/v1 per line; add lines while it runs
    # [--limit 200] [--ids-file subset.txt] [--bench --bench-out b.json]  for tests
"""
from __future__ import annotations

import argparse, asyncio, base64, difflib, json, os, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cit_pipeline.cot import PRETENDED_SYSTEM  # noqa: E402

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI  # noqa: E402
import httpx  # noqa: E402


class ServerDown(Exception):
    pass

V = Path(os.environ.get("CIT_DATA_ROOT", "data/cit") + "/procthor_v11")
REL = Path(os.environ.get("FLOWBENCH_RELEASE_ROOT", "data/flowbench_release") + "/sim/train")
COT = V / "cot"
LOG = COT / "_c0_log.jsonl"
MODEL = "Qwen3.5-9B"

BANNED = re.compile(r"(?i)\b(after[- ]image|second image|after photo|after frame|after picture|"
                    r"the after view|reference (?:cot|reasoning|text)|warm cot|private hint|"
                    r"the hint|hints? (?:say|said|suggest)|was shown|i was given|ground[- ]truth)\b")


AXIS = [(re.compile(r"(?i)(along the camera backward axis|backward along the camera axis|backward along the camera's axis)"),
         r"\1 (toward the camera)"),
        (re.compile(r"(?i)(along the camera forward axis|forward along the camera axis|forward along the camera's axis)"),
         r"\1 (away from the camera)")]


def explain_axes(s: str) -> str:
    """Spell out the camera-axis convention inline; the VLM otherwise reads 'backward' as 'deeper in'."""
    for pat, rep in AXIS:
        s = pat.sub(rep, s)
    return s


def user_prompt(question: str, move: str, answer: str, hint: str) -> str:
    move, question = explain_axes(move), explain_axes(question)
    """polish_pretended_cot prompt (cit_pipeline/cot.py) verbatim, plus two lines: the camera-axis
    convention (c_1 sometimes reads 'backward' as away from the camera) and 'own words' (c_0 copied c_1)."""
    return (
        "Write a pretended visual chain-of-thought for a student agent.\n"
        "The agent will only ever see the BEFORE image and the move command "
        "(the after image is counterfactual imagination, not shown).\n"
        "Constraints:\n"
        "- Keep the SAME final answer.\n"
        "- Reason from the before image + move: imagine where the object goes "
        "and what that implies for the question.\n"
        "- You may use the reference warm CoT below as private hints, but do "
        "NOT say you looked at an after image or at a reference CoT.\n"
        "- Never mention 'second image', 'after photo', or that an after frame "
        "was shown to you.\n"
        "- 4–8 short sentences, then a final line: Answer: <same answer>\n"
        # two lines added to the pipeline prompt (c_0 review, 2026-10-19):
        "- Directions: 'forward along the camera axis' moves the object AWAY from the camera, "
        "'backward along the camera axis' moves it TOWARD the camera.\n"
        "- Write your own reasoning in your own words; do not copy sentences from the reference.\n\n"
        f"Move command: {move}\n"
        f"Question: {question}\n"
        f"Ground-truth answer (must match): {answer}\n\n"
        f"Reference warm CoT (private hints only):\n{hint}\n"
    )


def norm_answer(s: str, cat: str):
    s = s.strip().lower().rstrip(".").replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    if cat == "closest_three":
        return frozenset(x.strip().removeprefix("and ").strip() for x in re.split(r",|\band\b", s) if x.strip())
    return s


def last_answer(text: str):
    m = re.findall(r"(?im)^\s*\**answer\**\s*:\s*(.+?)\s*$", text)
    if not m:
        m = re.findall(r"(?i)\banswer\s*:\s*(.+)", text)
    return m[-1].strip() if m else None


AWAY = re.compile(r"(?i)\b(away from the camera|farther from the camera|further from the camera|away from the viewer|"
                  r"deeper into the (?:room|scene)|farther into the (?:room|scene)|further into the (?:room|scene))\b")
TOWARD = re.compile(r"(?i)\b(toward(?:s)? the camera|closer to the camera|nearer to the camera|"
                    r"toward(?:s)? the viewer|into the foreground)\b")


def axis_of(text: str):
    t = text.lower()
    back = "backward axis" in t or "backward along the camera" in t
    fwd = "forward axis" in t or "forward along the camera" in t
    return "back" if back and not fwd else ("fwd" if fwd and not back else None)


def check(reply: str, want: str, cat: str, move: str = "", hint: str = "", choices=None, question: str = ""):
    probs = []
    mv = move.lower()
    if cat == "feasibility":            # the reasoning is about the probed move in the question
        a_m, a_q = axis_of(move), axis_of(question)
        mv = {"back": "backward axis", "fwd": "forward axis"}.get(a_q, "") if a_m in (None, a_q) else ""
    if "backward axis" in mv and "forward axis" not in mv and AWAY.search(reply):
        probs.append(f"direction:{AWAY.search(reply).group(0)}")
    if "forward axis" in mv and "backward axis" not in mv and TOWARD.search(reply):
        probs.append(f"direction:{TOWARD.search(reply).group(0)}")
    if hint:
        ratio = difflib.SequenceMatcher(None, hint.lower().split(), reply.lower().split()).ratio()
        if ratio > 0.80:
            probs.append(f"copy:{ratio:.2f}")
    got = last_answer(reply)
    if got is not None:                                   # "(C) the plate" / "C" -> the option text
        m = re.match(r"^\(?([A-F])\)?[.:]?\s*(.*)$", got.strip())
        if m and choices and m.group(1) in choices and (not m.group(2) or len(got.strip()) <= 3):
            got = choices[m.group(1)]
        elif m and m.group(2) and re.match(r"^\(?[A-F]\)", got.strip()):
            got = m.group(2)
    if got is None:
        probs.append("no_answer_line")
    elif norm_answer(got, cat) != norm_answer(want, cat):
        probs.append(f"answer_mismatch:{got[:40]}")
    if BANNED.search(reply):
        probs.append(f"banned:{BANNED.search(reply).group(0)}")
    nw = len(reply.split())
    if not 25 <= nw <= 260:
        probs.append(f"length:{nw}")
    return probs


def load_work(ids_file: Path | None, retry_failed: bool = False):
    allow = (ids_file or V / "filters/som_cot_ids.txt").read_text().split()
    want = set(allow)
    nat = {}
    with open(REL / "natural/dataset.jsonl") as f:
        for line in f:
            rid = line[8:line.index('"', 8)]
            if rid in want:
                nat[rid] = json.loads(line)
    done = set()
    if LOG.exists():
        for line in open(LOG):
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if rec["status"] == "ok" or not retry_failed:
                done.add(rec["id"])
    work, missing = [], 0
    for rid in allow:
        if rid in done:
            continue
        r = nat.get(rid)
        warm_p = COT / f"{rid}_warm.txt"
        if r is None or not warm_p.exists():
            missing += 1
            continue
        warm = warm_p.read_text().replace("<|endoftext|>", "").strip()
        ans = last_answer(warm)
        if not ans:
            missing += 1
            continue
        choices = {L: r[f"choice_{L}"] for L in "ABCDEF" if r.get(f"choice_{L}")}
        q = r["question"]
        if r["category"] in ("closer_to", "closest_three"):     # the options are the candidates
            q += " Options: " + "  ".join(f"({L}) {t}" for L, t in choices.items())
        work.append(dict(id=rid, cat=r["category"], move=r["motion_command"], choices=choices,
                         question=q, answer=ans, hint=warm,
                         image=str(REL / "natural" / r["image_before"])))
    return work, len(done), missing


def b64(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def sanitize_hint(hint: str, move: str) -> str:
    """Drop c_1 sentences whose stated direction contradicts the camera-axis motion."""
    mv = move.lower()
    bad = AWAY if ("backward axis" in mv and "forward axis" not in mv) else (
        TOWARD if ("forward axis" in mv and "backward axis" not in mv) else None)
    if bad is None:
        return hint
    sents = re.split(r"(?<=[.!?])\s+", hint)
    return " ".join(x for x in sents if not bad.search(x))


async def one(client, item, args, stats):
    def msgs_for(hint):
        return [{"role": "system", "content": PRETENDED_SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(item["image"])}},
                                             {"type": "text", "text": user_prompt(item["question"], item["move"],
                                                                                  item["answer"], hint)}]}]
    msgs = msgs_for(item["hint"])
    tried = []
    for attempt in range(args.max_attempts):
        if attempt == args.max_attempts - 1 and item["cat"] != "feasibility":
            clean = sanitize_hint(item["hint"], item["move"])
            if clean != item["hint"]:
                msgs = msgs_for(clean)            # last try: c_1 without its wrong-direction sentences
        temp = [0.3, 0.6, 0.8, 0.9][min(attempt, 3)]
        t0 = time.time()
        try:
            r = await client.chat.completions.create(
                model=MODEL, messages=msgs, max_tokens=700, temperature=temp, top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        except (APIConnectionError, APITimeoutError) as e:
            raise ServerDown(str(e)) from e
        except Exception as e:  # noqa: BLE001
            tried.append({"error": str(e)[:200]})
            await asyncio.sleep(2 + attempt * 3)
            continue
        text = (r.choices[0].message.content or "").strip()
        stats["tok_in"] += r.usage.prompt_tokens
        stats["tok_out"] += r.usage.completion_tokens
        stats["lat"].append(time.time() - t0)
        probs = check(text, item["answer"], item["cat"], item["move"], item["hint"], item["choices"], item["question"])
        tried.append({"temp": temp, "problems": probs})
        if not probs:
            if not args.bench:
                tmp = COT / f".{item['id']}_pretended.tmp"
                tmp.write_text(text + "\n")
                os.replace(tmp, COT / f"{item['id']}_pretended.txt")
            rec = dict(id=item["id"], status="ok", attempts=len(tried), tries=tried, model=MODEL)
            if args.bench:
                rec.update(text=text, c1=item["hint"], question=item["question"], move=item["move"],
                           answer=item["answer"], image=item["image"])
            return rec
    rec = dict(id=item["id"], status="failed", attempts=len(tried), tries=tried, model=MODEL)
    if args.bench:
        rec.update(c1=item["hint"], question=item["question"], move=item["move"], answer=item["answer"])
    return rec


async def healthy(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(url.rsplit("/v1", 1)[0] + "/health")
            return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def read_servers(args):
    if args.servers_file and os.path.exists(args.servers_file):
        return [l.split("#")[0].strip() for l in open(args.servers_file) if l.split("#")[0].strip()]
    return [s for s in (args.servers or "").split(",") if s]


async def main_async(args):
    work, n_done, n_missing = load_work(Path(args.ids_file) if args.ids_file else None, args.retry_failed)
    if args.limit:
        random.Random(args.seed).shuffle(work)
        work = work[: args.limit]
    print(f"work {len(work)} (already done {n_done}, skipped no-input {n_missing}); "
          f"concurrency {args.concurrency}/server", flush=True)
    q: asyncio.Queue = asyncio.Queue()
    for it in work:
        q.put_nowait(it)
    total = len(work)
    stats = {"ok": 0, "failed": 0, "tok_in": 0, "tok_out": 0, "lat": [], "t0": time.time(), "inflight": 0}
    logf = None if args.bench else open(LOG, "a")
    out_bench = []
    active: dict[str, int] = {}          # server -> live workers

    async def worker(url, client):
        active[url] = active.get(url, 0) + 1
        try:
            while True:
                try:
                    it = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                stats["inflight"] += 1
                try:
                    rec = await one(client, it, args, stats)
                except ServerDown:
                    q.put_nowait(it)          # give the item back; this server is gone
                    return
                finally:
                    stats["inflight"] -= 1
                stats[rec["status"]] += 1
                if logf:
                    logf.write(json.dumps(rec) + "\n"); logf.flush()
                else:
                    out_bench.append(rec)
        finally:
            active[url] -= 1

    async def manager():
        tasks = []
        while True:
            for url in read_servers(args):
                if active.get(url, 0) == 0 and not q.empty() and await healthy(url):
                    client = AsyncOpenAI(base_url=url, api_key="EMPTY", timeout=900, max_retries=0)
                    tasks += [asyncio.create_task(worker(url, client)) for _ in range(args.concurrency)]
                    print(f"[{time.strftime('%H:%M:%S')}] + server {url} ({args.concurrency} workers)", flush=True)
            if q.empty() and stats["inflight"] == 0:
                await asyncio.gather(*tasks, return_exceptions=True)
                return
            await asyncio.sleep(30)

    async def reporter():
        last_n, last_t = 0, time.time()
        while True:
            await asyncio.sleep(args.report_every)
            el = time.time() - stats["t0"]
            n = stats["ok"] + stats["failed"]
            inst = (n - last_n) / (time.time() - last_t)
            last_n, last_t = n, time.time()
            left = total - n
            lat = sorted(stats["lat"][-500:])
            live = sum(1 for v in active.values() if v > 0)
            print(f"[{time.strftime('%H:%M:%S')}] done {n}/{total} (ok {stats['ok']}, failed {stats['failed']}) "
                  f"{inst:.2f} req/s now ({n / el:.2f} avg), out {stats['tok_out'] / el:.0f} tok/s, "
                  f"p50 lat {lat[len(lat) // 2] if lat else 0:.1f}s, servers {live}; "
                  f"ETA {left / inst / 3600 if inst else float('nan'):.2f} h", flush=True)

    rep = asyncio.create_task(reporter())
    await manager()
    rep.cancel()
    el = time.time() - stats["t0"]
    n = stats["ok"] + stats["failed"]
    print(f"FINISHED {n} in {el:.0f}s: {n / el:.2f} req/s, in {stats['tok_in'] / el:.0f} tok/s, "
          f"out {stats['tok_out'] / el:.0f} tok/s; ok {stats['ok']} failed {stats['failed']}", flush=True)
    if args.bench_out:
        json.dump(dict(concurrency=args.concurrency, servers=read_servers(args), n=n, seconds=el, req_s=n / el,
                       out_tok_s=stats["tok_out"] / el, in_tok_s=stats["tok_in"] / el, ok=stats["ok"],
                       failed=stats["failed"], recs=out_bench), open(args.bench_out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", default=None, help="comma list (or use --servers-file)")
    ap.add_argument("--servers-file", default=None, help="one URL per line; re-read every 30 s")
    ap.add_argument("--concurrency", type=int, default=64, help="in-flight requests per server")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--report-every", type=int, default=60)
    ap.add_argument("--bench", action="store_true", help="do not write outputs or the log")
    ap.add_argument("--retry-failed", action="store_true", help="redo ids whose last logged status is failed")
    ap.add_argument("--bench-out", default=None)
    asyncio.run(main_async(ap.parse_args()))
