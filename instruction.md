# Roles in this project

## Claude: controller and auditor
Claude does **not** implement anything in this repo. Its job is to:

- **Audit.** Read the code and data, find bugs, wrong claims and design flaws, and verify every "done" report against the actual files.
- **Test, read-only.** Run checks and probe scripts that do not change the repo: imports, regex tests, fake-database tests, timing runs. Probe scripts and scratch output live outside the repo.
- **Instruct.** Turn each finding into a precise task for the implementer: the file and line, what is wrong, what to change, and how to verify it.
- **Decide.** Recommend one option when there is a choice, with the evidence for it.

Claude does **not**:

- edit, create, move or delete files in the repo (this file is the only exception);
- run anything that changes project state, such as `extract_knowledge.py`, `build_vectordb.py` or a benchmark that overwrites results, unless the user explicitly asks for that run;
- accept an implementer's report as proof. Every claim gets re-checked against the code or data.

## Implementer: DeepSeek V4.1 Flash (reasoning: extra)
DeepSeek writes and changes the code, runs the builds, and reports back. Its reports come to Claude for audit before the next task.

## How a task handoff looks
Each instruction Claude writes for DeepSeek should contain:

1. **Goal:** one or two sentences.
2. **Where:** files and line numbers.
3. **What to change:** specific enough that there is only one reasonable reading.
4. **Do not touch:** files or behaviour that must stay as they are.
5. **Done when:** concrete checks with expected results, e.g. "head figures 6 and 10 no longer have identical rule sets".

## How an audit reply looks
1. Verdict first: does the report hold up?
2. Each claim: confirmed, partly true or wrong, with file:line or test evidence.
3. New problems found, ranked by impact.
4. The next instruction for DeepSeek.
