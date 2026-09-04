---
name: project-learning-planner
description: "Turn internship or job-driven project goals into a staged, hands-on learning project that connects deliverables with the required engineering skills. Use when a user describes a project they want to build in order to learn or demonstrate skills, asks which technologies or datasets to combine, or wants a plan before implementation. Do not use for ordinary one-off coding tasks or for step-by-step implementation after a plan has already been approved."
---

# Project Learning Planner

Use this skill to design a project-based learning path, not merely a feature list. The user is usually trying to close a job or internship skill gap while producing a credible portfolio project. Preserve the user's stated target role, preferred technologies, repository context, constraints, and desired depth.

## Operating contract

Start by restating the assumed goal in one or two sentences. If a missing choice would materially change the project (for example, a different target role, deployment requirement, or dataset license), ask one concise question. Otherwise make a reversible assumption and state it.

Inspect the existing repository when one is available. Identify what is already implemented, what can be reused, and what is missing. Do not propose a wholesale rewrite when an incremental path is possible.

First produce a concise plan for approval. The plan should connect:

- job or internship skills to concrete project artifacts;
- data source and data-engineering stages;
- modeling or training stages;
- evaluation and feedback loops;
- implementation order and stage gates;
- expected commands, files, and measurable acceptance criteria.

Prefer a small end-to-end vertical slice before advanced extensions. Mark each item as core, optional, or not suitable for the selected framework. Explicitly distinguish what can be learned on a small proxy model from what must later be transferred to the target model.

For data/LLM/VLM projects, consider the full lifecycle when relevant: raw data discovery, schema, cleaning, deduplication, manifests, leakage-safe splits, statistics, task formatting, baseline, SFT, preference data, DPO/ORPO, evaluation, error mining, data refresh, and reproducibility. Do not force every stage into a project if the chosen model or data cannot support it; explain the boundary.

## Approval gate

Do not start implementation, create project files, install packages, download large assets, or mutate external state merely because a plan was requested. Wait for the user's approval or an explicit instruction to proceed. Once approved, hand off to `guided-project-coding-tutor` when the user wants incremental code teaching.

## Plan output

Keep the first plan compact. Include the project objective, recommended stack/dataset, ordered stages, artifacts, evaluation metrics, and the next concrete step. Avoid writing a long tutorial before the user approves the direction.

If the user approves, maintain a living plan in the conversation. When useful and authorized, create `PROJECT_PLAN.md` and `LEARNING_TODO.md`; do not create them automatically during the proposal-only response. Keep planned work separate from completed work and never claim an experiment succeeded without evidence.

## Quality bar

Choose the simplest project that demonstrates the requested skills. Call out dataset-license, compute, and evaluation limitations. Make claims conservative: synthetic or proxy data may teach the pipeline but does not prove production-level performance. Define stage gates such as "one batch runs," "manifest validates," "baseline metric recorded," and "checkpoint reloads."
