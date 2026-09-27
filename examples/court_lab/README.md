# Court Lab: a cross-examination lab with a known truth

<div dir="rtl">

## The idea in one sentence

In real trials, nobody knows what a witness hid, so you can't learn from them automatically. In a simulation we plant the secrets ourselves, so every session can be scored automatically: what was revealed, on which question, and with which tactic. That's labeled data. The lab distills it into a "playbook" and gives it to the next generation of examiners.

## How it works

1. **Case** (`cases/*.json`): public background, evidence for the lawyer, facts the witness may tell, and **secrets with IDs** (`S1`, `S2`...) that the witness is told to hide.
2. **Three roles, separate information:**
   - **Examiner:** sees only the background and the evidence. Asks one question per turn through the `ask_witness` tool (question + tactic + evidence shown).
   - **Witness:** sees its secrets and hides them. When shown concrete evidence, it may reluctantly admit part of the truth.
   - **Analyst:** gets the list of secrets and every question/answer pair, and marks what was revealed with the `record_findings` tool.
3. **Scoring:** reveal rate, the turn on which each secret came out, and which tactic led to it.
4. **Learning:** `distill_playbook` computes how successful each tactic was and collects questions that worked. The next generation receives these lessons in its instructions. `summary.json` compares the generations.

Each role can use a different model (`Roles`): for example a Hebrew-native model for the witness and a cheap model for the analyst.

## Running

</div>

```shell
# In-process with a GGUF file (CPU/laptop, no server):
pip install llama-cpp-python
python examples/court_lab/court_lab.py --gguf gpt-oss-20b-MXFP4.gguf --generations 2 --turns 5

# Or any OpenAI-compatible server (Ollama, vLLM, Groq, DeepInfra...):
python examples/court_lab/court_lab.py --base-url https://api.groq.com/openai/v1 \
    --model openai/gpt-oss-20b --api-key $GROQ_API_KEY --generations 3 --sessions 10
```

<div dir="rtl">

Output: `sessions.jsonl` (every question, answer and label), `summary.json` (per-generation comparison) and `playbook.txt` (the latest lessons).

## Notes

- The analyst is also a model, so the labels can contain errors. Check a sample of them by hand.
- One or two sessions per generation isn't statistically significant. For a real conclusion, run dozens of sessions per generation (on Groq that costs a few cents).
- The goal is training lawyers to expose concealment and finding weak points in your own case before trial, not teaching witnesses to hide the truth.

</div>
