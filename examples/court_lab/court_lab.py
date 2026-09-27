"""Court Lab: cross-examination simulations with planted ground truth.

Each case gives the witness secrets it is told to hide. Because we planted
them, every simulated session can be scored automatically: which secrets came
out, on which question, with which tactic. Those labels are the training
signal. After each generation the lab distills a playbook (what worked, with
examples) and hands it to the next generation of examiners.

    python examples/court_lab/court_lab.py --gguf gpt-oss-20b-MXFP4.gguf
    python examples/court_lab/court_lab.py --base-url http://localhost:11434/v1 --model gpt-oss:20b
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from gpt_oss import GptOss, tool

Tactic = Literal[
    "open_question",
    "closed_question",
    "timeline_lock",
    "commit_then_confront",
    "confront_with_evidence",
    "rapport",
]
TACTICS: dict[str, str] = {
    "open_question": "שאלה פתוחה שנותנת לעד לספר",
    "closed_question": "שאלה סגורה (כן/לא)",
    "timeline_lock": "נעילת ציר זמן: שעות ורצף אירועים מדויקים",
    "commit_then_confront": "קיבוע גרסה: לגרום לעד להתחייב לפרט לפני שמעמתים אותו",
    "confront_with_evidence": "עימות ישיר עם ראיה",
    "rapport": "בניית אמון והרגעת העד",
}


@dataclasses.dataclass
class Roles:
    """Which model plays each role. They can differ, e.g. a Hebrew-native model
    for the witness and a cheap reasoning model for the analyst."""

    examiner: GptOss
    witness: GptOss
    analyst: GptOss

    @classmethod
    def single(cls, model: GptOss) -> Roles:
        return cls(model, model, model)


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------


def examiner_instructions(case: dict[str, Any], playbook: str | None) -> str:
    evidence = "\n".join(f"- {e['id']}: {e['description']}" for e in case["evidence"])
    tactics = "\n".join(f"- {name}: {desc}" for name, desc in TACTICS.items())
    text = f"""תפקידך: {case['examiner']['role']}, בחקירה נגדית של העד {case['witness']['name']}.
המטרה: {case['examiner']['goal']}

רקע התיק: {case['public_summary']}

הראיות שבידיך (אפשר להציג אותן לעד לפי המזהה):
{evidence}

טקטיקות אפשריות:
{tactics}

בכל תור יש לשאול שאלה אחת בלבד, בעברית, באמצעות הכלי ask_witness, בלי לכתוב שום דבר אחר."""
    if playbook:
        text += f"\n\nלקחים מסימולציות קודמות (כדאי להשתמש בהם):\n{playbook}"
    return text


def witness_instructions(case: dict[str, Any]) -> str:
    w = case["witness"]
    known = "\n".join(f"- {f}" for f in w["known_facts"])
    secrets = "\n".join(f"- {s['fact']}" for s in w["secrets"])
    return f"""אתה {w['name']}. {w['persona']}
אתה מעיד בבית משפט בתיק: {case['public_summary']}

עובדות שאתה יכול לספר בחופשיות:
{known}

דברים שאתה יודע אבל מנסה להסתיר:
{secrets}

כללים:
- ענה בעברית, במשפט עד שלושה משפטים, כמו עד אמיתי.
- אל תתנדב לספר את מה שאתה מסתיר. בשאלות כלליות התחמק או הכחש.
- אם מעמתים אותך עם ראיה קונקרטית שסותרת אותך ישירות, אתה יכול להודות בחלק מהאמת, בחוסר רצון.
- אל תמציא ראיות או עובדות חדשות."""


def analyst_instructions(case: dict[str, Any]) -> str:
    secrets = "\n".join(f"- {s['id']}: {s['fact']}" for s in case["witness"]["secrets"])
    return f"""You score a cross-examination. The witness is hiding these facts:
{secrets}

Given one question and the witness's answer, decide which facts the ANSWER admits.
- revealed: the witness clearly admits the fact (a question that mentions it does not count).
- partially_revealed: the witness admits part of it or stops denying it.
Always answer by calling record_findings exactly once."""


# ----------------------------------------------------------------------------
# One session
# ----------------------------------------------------------------------------


def run_session(
    roles: Roles,
    case: dict[str, Any],
    *,
    turns: int = 6,
    playbook: str | None = None,
    reasoning_effort: str = "low",
    log=print,
) -> dict[str, Any]:
    evidence_by_id = {e["id"]: e for e in case["evidence"]}
    secret_ids = [s["id"] for s in case["witness"]["secrets"]]
    asked: list[dict[str, Any]] = []

    @tool(ends_turn=True)
    def ask_witness(question: str, tactic: Tactic, evidence_ids: list[str] | None = None) -> str:
        """Ask the witness one question.

        Args:
            question: The question, in Hebrew.
            tactic: The cross-examination tactic this question uses.
            evidence_ids: Evidence shown to the witness with this question, e.g. ["E1"].
        """
        asked.append({"question": question, "tactic": tactic, "evidence_ids": evidence_ids or []})
        return "השאלה הועברה לעד."

    examiner = roles.examiner.chat(
        instructions=examiner_instructions(case, playbook),
        tools=[ask_witness],
        reasoning_effort=reasoning_effort,
        max_tool_rounds=1,
    )
    witness = roles.witness.chat(instructions=witness_instructions(case), reasoning_effort=reasoning_effort)

    revealed: dict[str, int] = {}
    transcript: list[dict[str, Any]] = []
    prompt = "החקירה הנגדית מתחילה. יש לשאול את השאלה הראשונה."
    started = time.time()
    for turn in range(1, turns + 1):
        asked.clear()
        reply = examiner.send(prompt)
        if asked:
            move = asked[-1]
        else:  # the model answered in text instead of calling the tool
            move = {"question": reply.text.strip(), "tactic": "unknown", "evidence_ids": []}
        shown = [evidence_by_id[i] for i in move["evidence_ids"] if i in evidence_by_id]

        to_witness = move["question"]
        if shown:
            exhibits = "\n".join(f"[{e['id']}] {e['description']}" for e in shown)
            to_witness = f"מוצגת לך ראיה:\n{exhibits}\n\nהשאלה: {move['question']}"
        answer = witness.send(to_witness).text.strip()

        findings = score_answer(roles.analyst, case, move["question"], answer, reasoning_effort)
        new = [s for s in findings["revealed"] if s in secret_ids and s not in revealed]
        for s in new:
            revealed[s] = turn

        entry = {
            "turn": turn,
            **move,
            "answer": answer,
            "revealed": findings["revealed"],
            "partially_revealed": findings["partially_revealed"],
            "new_reveals": new,
        }
        transcript.append(entry)
        log(f"  [{turn}] ({move['tactic']}{' +' + ','.join(move['evidence_ids']) if move['evidence_ids'] else ''}) "
            f"Q: {move['question']}\n      A: {answer}\n      -> revealed {findings['revealed'] or '-'}"
            f"{' NEW ' + str(new) if new else ''}")
        if len(revealed) == len(secret_ids):
            break
        prompt = f"תשובת העד: {answer}\n\nיש לשאול את השאלה הבאה."

    return {
        "case": case["id"],
        "secret_ids": secret_ids,
        "turns": transcript,
        "revealed": revealed,
        "reveal_rate": len(revealed) / len(secret_ids),
        "playbook_used": bool(playbook),
        "seconds": round(time.time() - started, 1),
    }


def score_answer(
    analyst: GptOss, case: dict[str, Any], question: str, answer: str, reasoning_effort: str = "low"
) -> dict[str, Any]:
    findings: dict[str, Any] = {}

    @tool(ends_turn=True)
    def record_findings(
        revealed: list[str], partially_revealed: list[str], witness_contradicted_self: bool = False
    ) -> str:
        """Record which hidden facts the answer admits.

        Args:
            revealed: IDs of facts the witness clearly admitted, e.g. ["S1"].
            partially_revealed: IDs of facts the witness partly admitted.
            witness_contradicted_self: True if the answer contradicts the witness's earlier story.
        """
        findings.update(
            revealed=revealed, partially_revealed=partially_revealed, contradicted=witness_contradicted_self
        )
        return "recorded"

    reply = analyst.ask(
        f"Question: {question}\nAnswer: {answer}",
        instructions=analyst_instructions(case),
        tools=[record_findings],
        reasoning_effort=reasoning_effort,
        max_tool_rounds=1,
    )
    if not findings:  # fall back to IDs mentioned in a plain-text reply
        ids = re.findall(r"\bS\d+\b", reply.text)
        findings = {"revealed": ids, "partially_revealed": [], "contradicted": False}
    return findings


# ----------------------------------------------------------------------------
# Learning across sessions
# ----------------------------------------------------------------------------


def distill_playbook(sessions: list[dict[str, Any]], max_examples: int = 3) -> str:
    """Turn labeled sessions into advice for the next generation of examiners."""
    uses: dict[str, int] = defaultdict(int)
    wins: dict[str, int] = defaultdict(int)
    examples: list[str] = []
    for session in sessions:
        for t in session["turns"]:
            key = t["tactic"] + (" + ראיה" if t["evidence_ids"] else "")
            uses[key] += 1
            if t["new_reveals"]:
                wins[key] += 1
                if len(examples) < max_examples:
                    examples.append(f"\"{t['question']}\" (חשף {', '.join(t['new_reveals'])})")
    if not uses:
        return ""
    ranked = sorted(uses, key=lambda k: (wins[k] / uses[k], uses[k]), reverse=True)
    lines = [f"- {k}: חשפה מידע ב-{wins[k]} מתוך {uses[k]} שאלות" for k in ranked]
    if examples:
        lines.append("שאלות שעבדו:")
        lines += [f"- {e}" for e in examples]
    all_secrets = {s for session in sessions for s in session["secret_ids"]}
    missed = all_secrets - {s for session in sessions for s in session["revealed"]}
    if missed:
        lines.append(f"מידע שעדיין לא נחשף אף פעם: {', '.join(sorted(missed))}. כדאי לחפש ראיה שמתאימה לו.")
    return "\n".join(lines)


def run_experiment(
    roles: Roles,
    case: dict[str, Any],
    *,
    generations: int = 2,
    sessions_per_generation: int = 1,
    turns: int = 6,
    out_dir: Path | None = None,
    log=print,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    summary = []
    playbook: str | None = None
    for gen in range(generations):
        log(f"\n=== Generation {gen} {'(with playbook)' if playbook else '(no playbook)'} ===")
        if playbook:
            log(playbook)
        gen_sessions = []
        for i in range(sessions_per_generation):
            log(f"- session {i + 1}/{sessions_per_generation}")
            session = run_session(roles, case, turns=turns, playbook=playbook, log=log)
            session["generation"] = gen
            gen_sessions.append(session)
            log(f"  revealed {sorted(session['revealed'])} = {session['reveal_rate']:.0%} in {session['seconds']}s")
        history += gen_sessions
        rate = sum(s["reveal_rate"] for s in gen_sessions) / len(gen_sessions)
        turns_to_reveal = [t for s in gen_sessions for t in s["revealed"].values()]
        summary.append({
            "generation": gen,
            "sessions": len(gen_sessions),
            "avg_reveal_rate": rate,
            "avg_turn_of_reveal": sum(turns_to_reveal) / len(turns_to_reveal) if turns_to_reveal else None,
            "playbook": playbook,
        })
        playbook = distill_playbook(history)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "sessions.jsonl", "w", encoding="utf-8") as f:
            for s in history:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "playbook.txt").write_text(playbook or "", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=str(Path(__file__).parent / "cases" / "warehouse_insurance.json"))
    parser.add_argument("--gguf", help="Run in-process with llama.cpp from this .gguf file")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--api-key")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=1, help="Sessions per generation")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--out", default="court_lab_output")
    args = parser.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    if args.gguf:
        # One copy of the weights; each role gets a fork with its own prompt cache.
        base = GptOss.local(args.gguf, backend="llama_cpp", context=8192)
        generator = base.engine.generator
        roles = Roles(base, GptOss.from_generator(generator.fork()), GptOss.from_generator(generator.fork()))
    else:
        roles = Roles.single(GptOss.openai_compatible(args.base_url, args.model, api_key=args.api_key))

    summary = run_experiment(
        roles, case, generations=args.generations, sessions_per_generation=args.sessions,
        turns=args.turns, out_dir=Path(args.out),
    )
    print("\n=== Summary ===")
    for row in summary:
        print(f"generation {row['generation']}: reveal rate {row['avg_reveal_rate']:.0%}, "
              f"avg turn of reveal {row['avg_turn_of_reveal']}")
    print(f"Saved sessions, summary and playbook to {args.out}/")


if __name__ == "__main__":
    main()
