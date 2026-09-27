import json
import sys
from pathlib import Path

import pytest

from gpt_oss.sdk import GptOss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "court_lab"))
import court_lab  # noqa: E402

CASE = json.loads(
    (Path(court_lab.__file__).parent / "cases" / "warehouse_insurance.json").read_text(encoding="utf-8")
)


class RoleScript:
    """Replays harmony completions for one role."""

    def __init__(self, encoding, completions):
        self.encoding = encoding
        self.completions = list(completions)
        self.prompts = []

    def generate(self, prompt_tokens, stop_tokens, temperature=1.0, max_tokens=0):
        self.prompts.append(self.encoding.decode(list(prompt_tokens)))
        for token in self.encoding.encode(self.completions.pop(0), allowed_special="all"):
            yield token
            if token in stop_tokens:
                return


def ask(question, tactic, evidence=()):
    args = json.dumps({"question": question, "tactic": tactic, "evidence_ids": list(evidence)}, ensure_ascii=False)
    return f"<|channel|>commentary to=functions.ask_witness <|constrain|>json<|message|>{args}<|call|>"


def answer(text):
    return f"<|channel|>final<|message|>{text}<|return|>"


def findings(*revealed):
    args = json.dumps({"revealed": list(revealed), "partially_revealed": []})
    return f"<|channel|>commentary to=functions.record_findings <|constrain|>json<|message|>{args}<|call|>"


@pytest.fixture
def roles(harmony_encoding):
    def make(examiner, witness, analyst):
        scripts = [RoleScript(harmony_encoding, c) for c in (examiner, witness, analyst)]
        return court_lab.Roles(*(GptOss.from_generator(s) for s in scripts)), scripts

    return make


def test_session_scores_reveals_against_planted_secrets(roles):
    r, (examiner, witness, analyst) = roles(
        [ask("מתי יצאת מהמחסן?", "timeline_lock"), ask("איך אתה מסביר את זה?", "confront_with_evidence", ["E1"])],
        [answer("ב-23:00 כמו תמיד."), answer("טוב... יצאתי קצת לפני, בסביבות 22:30.")],
        [findings(), findings("S1")],
    )
    session = court_lab.run_session(r, CASE, turns=2, log=lambda *_: None)

    assert session["revealed"] == {"S1": 2}
    assert session["reveal_rate"] == pytest.approx(1 / 3)
    assert [t["tactic"] for t in session["turns"]] == ["timeline_lock", "confront_with_evidence"]
    # Hidden facts reach only the witness; the examiner only sees public info and evidence.
    assert "לא נעל את הדלת האחורית" in witness.prompts[0]
    assert "לא נעל את הדלת האחורית" not in examiner.prompts[0]
    # Evidence shown by the examiner is put in front of the witness.
    assert "יומן בקרת הכניסה" in witness.prompts[1]
    # The examiner hears the witness's answer on its next turn.
    assert "ב-23:00 כמו תמיד." in examiner.prompts[1]


def test_playbook_is_learned_from_labeled_sessions():
    sessions = [{
        "secret_ids": ["S1", "S2", "S3"],
        "revealed": {"S1": 2},
        "turns": [
            {"tactic": "open_question", "evidence_ids": [], "new_reveals": [], "question": "ספר"},
            {"tactic": "confront_with_evidence", "evidence_ids": ["E1"], "new_reveals": ["S1"], "question": "הסבר את היומן"},
        ],
    }]
    playbook = court_lab.distill_playbook(sessions)
    lines = playbook.splitlines()
    assert lines[0] == "- confront_with_evidence + ראיה: חשפה מידע ב-1 מתוך 1 שאלות"
    assert "\"הסבר את היומן\" (חשף S1)" in playbook
    assert "S2, S3" in playbook


def test_experiment_feeds_playbook_to_next_generation(roles, tmp_path):
    r, (examiner, witness, analyst) = roles(
        [ask("הסבר את היומן", "confront_with_evidence", ["E1"]), ask("שוב", "open_question")],
        [answer("יצאתי ב-22:30."), answer("אין לי מה להוסיף.")],
        [findings("S1"), findings()],
    )
    summary = court_lab.run_experiment(r, CASE, generations=2, turns=1, out_dir=tmp_path, log=lambda *_: None)

    assert [row["avg_reveal_rate"] for row in summary] == [pytest.approx(1 / 3), 0]
    assert "לקחים מסימולציות קודמות" not in examiner.prompts[0]
    assert "לקחים מסימולציות קודמות" in examiner.prompts[1]
    assert len((tmp_path / "sessions.jsonl").read_text(encoding="utf-8").splitlines()) == 2
