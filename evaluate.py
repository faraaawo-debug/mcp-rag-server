"""Evaluates the pipeline on a set of annotated questions (eval_set.json).

Usage:
  python evaluate.py --label baseline                    # full evaluation
  python evaluate.py --label test --retrieval-only       # retrieval only (~1 min, no LLM)
  python evaluate.py --label baseline --regrade results/baseline.json
      # grades already generated answers again, without regenerating them

Metrics:
- retrieval (fixed k, independent of the pipeline): right file within the first k excerpts,
  key sentence within the first k excerpts, MRR of the key sentence
- router: rate of correctly detected intents (n/a as long as there is no router)
- answers: correctness and faithfulness (LLM judge), correct refusals, mean generation time
The script also suggests a threshold for the server's reliability status.
"""
import argparse
import json
import re
import statistics

import config
import llm
from rag_utils import answer_question, is_refusal, search

# Prompts in English (language of the documents and questions) and enforced JSON output:
# a single verdict, readable without ambiguity.
JUDGE_CORRECTNESS = """You are grading an answer against a reference answer.
Question: {question}
Reference answer: {expected}
Proposed answer: {proposed}

Does the proposed answer contain the essential information of the reference answer, without contradicting it?
An answer that says the information is not available, or that misses the essential point, is NOT correct.
Reply with JSON only, exactly one of: {{"verdict": "YES"}} or {{"verdict": "NO"}}"""

# The judge gets the question and the excerpts labelled with their file, exactly like the
# generator: without them, it rejected faithful answers (a file citation it could not check,
# a short answer that made no sense without the question).
JUDGE_FAITHFULNESS = """You are checking whether an answer is supported by document excerpts.
Each excerpt starts with its source file name in square brackets.

Question: {question}

Excerpts:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the excerpts? Rephrasing or summarizing the excerpts is allowed.
Citing a source file name that appears in the excerpt labels is not a claim to check.
If at least one factual claim is not supported by the excerpts, the verdict is NO.
Reply with JSON only, exactly one of: {{"verdict": "YES"}} or {{"verdict": "NO"}}"""


def judge(prompt):
    """Asks the judge model a closed question.
    Returns (verdict, raw text); verdict is None if the output cannot be read."""
    raw = llm.chat([{"role": "user", "content": prompt}],
                   provider=config.JUDGE_PROVIDER, model=config.JUDGE_MODEL, json_mode=True)
    try:
        verdict = str(json.loads(raw).get("verdict", "")).strip().upper()
    except (json.JSONDecodeError, AttributeError):
        return None, raw
    return {"YES": True, "NO": False}.get(verdict), raw


def normalize(text):
    """Lowercase, no spaces or punctuation: the key sentence is found even if the PDF
    extraction glued or split words ("toshare" / "to share")."""
    return re.sub(r"[\W_]+", "", text.lower())


def load_eval_set():
    return json.loads(config.EVAL_SET_PATH.read_text(encoding="utf-8"))


def measure_retrieval(item):
    """Rank (1, 2, ...) of the right file and of the first excerpt containing the key
    sentence, among the first max(K_EVAL) excerpts. None if absent."""
    passages = search(item["question"], k=max(config.K_EVAL))
    key = normalize(item["key_sentence"])
    file_rank = next(
        (i + 1 for i, p in enumerate(passages) if p["source"] == item["expected_source"]), None)
    key_rank = next(
        (i + 1 for i, p in enumerate(passages)
         if p["source"] == item["expected_source"] and key in normalize(p["text"])), None)
    return file_rank, key_rank


def grade_row(row, item):
    """Adds the correctness and faithfulness verdicts to a result row."""
    if not item["answerable"]:
        return
    row["correct"], row["judge_correctness"] = judge(JUDGE_CORRECTNESS.format(
        question=item["question"], expected=item["expected_answer"], proposed=row["answer"]))
    if row["status"] == "refusal":
        row["faithful"], row["judge_faithfulness"] = None, None  # a refusal makes no claim
    else:
        context = "\n\n".join(f"[{p['source']}] {p['text']}" for p in row["passages"])
        row["faithful"], row["judge_faithfulness"] = judge(JUDGE_FAITHFULNESS.format(
            question=item["question"], context=context, answer=row["answer"]))


def suggest_threshold(rows):
    """Looks for the threshold that best separates good answers from bad ones."""
    candidates = [r for r in rows if r["overall_score"] is not None]
    if len(candidates) < 4:
        return None
    best = None
    for t in [x / 100 for x in range(50, 96)]:
        well_classified = sum(
            (r["overall_score"] >= t) == (r["correct"] is True and r["faithful"] is True) for r in candidates
        )
        rate = well_classified / len(candidates)
        if best is None or rate > best[1]:
            best = (t, rate)
    return best


def pct(values):
    values = list(values)
    return f"{100 * sum(values) / len(values):.0f} % ({sum(values)}/{len(values)})" if values else "n/a"


def summarize(label, rows, retrieval_only=False):
    answerable = [r for r in rows if r["answerable"]]
    summary = {"version": label, "questions": len(rows)}
    for k in config.K_EVAL:
        summary[f"file_found@{k}"] = pct(r["file_rank"] is not None and r["file_rank"] <= k for r in answerable)
        summary[f"key_sentence@{k}"] = pct(r["key_rank"] is not None and r["key_rank"] <= k for r in answerable)
    summary[f"mrr_key_sentence@{max(config.K_EVAL)}"] = round(
        statistics.mean(1 / r["key_rank"] if r["key_rank"] else 0 for r in answerable), 3)
    if retrieval_only:
        return summary

    unanswerable = [r for r in rows if not r["answerable"]]
    with_intent = [r for r in rows if r["detected_intent"] is not None]
    summary["correct_intents"] = pct(
        r["detected_intent"] == r["expected_intent"] for r in with_intent)
    summary["correctness"] = pct(r["correct"] is True for r in answerable)
    summary["faithfulness_given_answers"] = pct(
        r["faithful"] is True for r in answerable if r["status"] != "refusal")
    summary["correct_refusals"] = pct(r["correct_refusal"] for r in unanswerable)
    summary["mean_latency_s"] = round(statistics.mean(r["latency_s"] for r in rows), 2)
    summary["generator_model"] = f"{config.LLM_PROVIDER}/{config.LLM_MODEL}"
    summary["judge_model"] = f"{config.JUDGE_PROVIDER}/{config.JUDGE_MODEL}"
    # A refusal deliberately has no faithfulness verdict: it is not counted here
    summary["unreadable_verdicts"] = sum(r["correct"] is None for r in answerable) + sum(
        r["faithful"] is None for r in answerable if r["status"] != "refusal")
    threshold = suggest_threshold([r for r in answerable if r["status"] != "refusal"])
    if threshold:
        summary["suggested_threshold"] = threshold[0]
        summary["indicator_accuracy"] = f"{100 * threshold[1]:.0f} %"
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="latest", help="name of the evaluated version (e.g. baseline)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="only measures retrieval, without calling the LLM")
    parser.add_argument("--regrade", metavar="FILE",
                        help="grades the answers of a results file again, without regenerating them")
    args = parser.parse_args()

    eval_set = load_eval_set()
    if args.regrade:
        previous = json.loads(open(args.regrade, encoding="utf-8").read())["details"]
        if [r["question"] for r in previous] != [item["question"] for item in eval_set]:
            raise ValueError("The file to regrade does not match the current evaluation set.")
        if any("passages" not in r for r in previous if r["answerable"]):
            raise ValueError("This file does not contain the excerpts used: it cannot be regraded.")

    rows = []
    for n, item in enumerate(eval_set, 1):
        row = {
            "question": item["question"],
            "answerable": item["answerable"],
            "expected_intent": item["expected_intent"],
        }
        if item["answerable"]:
            row["file_rank"], row["key_rank"] = measure_retrieval(item)

        if not args.retrieval_only:
            if args.regrade:
                old = previous[n - 1]
                for key in ("detected_intent", "answer", "status", "overall_score",
                            "latency_s", "sources", "passages"):
                    row[key] = old[key]
            else:
                result = answer_question(item["question"])
                row.update({
                    "detected_intent": result.get("intent"),  # absent as long as there is no router
                    "answer": result["answer"],
                    "status": result["status"],
                    "overall_score": result["overall_score"],
                    "latency_s": result["latency_s"],
                    "sources": result["sources"],
                    # excerpts actually given to the LLM: allow regrading without regenerating
                    "passages": [{k: p[k] for k in ("source", "chunk", "similarity", "text")}
                                 for p in result["passages"]],
                })
            if item["answerable"]:
                grade_row(row, item)
            else:
                row["correct_refusal"] = is_refusal(row["answer"])
            print(f"[{n}/{len(eval_set)}] {item['question'][:60]} -> {row['status']}", flush=True)
        rows.append(row)

    summary = summarize(args.label, rows, args.retrieval_only)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / f"{args.label}.json"
    path.write_text(json.dumps({"summary": summary, "details": rows}, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    print("\n| Metric | Result |\n|---|---|")
    for key, value in summary.items():
        print(f"| {key} | {value} |")
    print(f"\nDetails saved in {path.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
