---
name: ibrobot-git-flow
description: "Handles Git commit and push workflow with Euler compliance. Use when user asks to 'git commit', 'git push', '提交代码', '推送代码', 'git status', 'commit -s', 'DCO sign-off', 'check git message', 'euler compliance', '符合规范'. Triggers for 'push to fork', 'commit changes', '提交', '检查提交信息', or when preparing code for submission."
---

# IB_Robot Git Workflow Guide

This skill automates code commit process for IB_Robot project (root repo with src directory) and its submodule `libs/lerobot`. The commit process enforces openEuler Embedded specification.

## Core Specifications

### 0. Commit Scope — Only Commit What You Changed

**Do NOT blindly `git add .` and commit everything.** Before staging, carefully inspect `git diff` and `git status` to ensure only files related to the current task are included.

**`libs/lerobot` exclusion rule:**
- `libs/lerobot` is managed via a **patch stack** (`third_party/patches/lerobot/`). Changes inside `libs/lerobot` must be exported as patches using the `ibrobot-lerobot-patch` skill — they should **never** be committed directly to the root repo via `git add libs/lerobot`.
- Therefore, during normal `git commit` workflow, **always exclude `libs/lerobot` from staging**. Use `git add` on specific files or paths, NOT `git add .` from the repo root.
- The only exception: the user explicitly asks to commit lerobot changes directly (e.g., "把 lerobot 的改动也提交了").

**Practical rule:**
1. Review `git status` output carefully.
2. Stage only the files that belong to the current task using `git add <specific-paths>`.
3. If `libs/lerobot` appears in `git status`, explicitly skip it with `git restore --staged libs/lerobot` or simply do not stage it.
4. If unrelated files (not part of the current task) appear modified, skip them too unless the user asks to include them.

### 1. Commit Message Format
Must strictly follow this structure with exactly one blank line between sections:

```
<area>: <subject>

<body>

<footer_tags>
```

- **Title (<area>: <subject>)**:
  - Format: `<module>: <brief description>` (e.g., `robot_interface: fix moveit crash`).
  - Length limit: Non-revert commits max 80 chars, revert commits max 102 chars.
  - Subject must have at least 2 words, no trailing punctuation.
  - Exactly one space after colon.
  - No Chinese characters allowed.
- **Body**:
  - Must provide detailed description explaining "why" and "what".
  - Each line max 100 characters (unless containing URL).
  - No Chinese characters allowed.
- **Footer (Tags)**:
  - Must include `Signed-off-by: Name <email>`.
  - `Signed-off-by` must be the last line.
  - Allowed tags: `Fixes`, `Closes`, `Co-developed-by`, `Link`.
  - Tags must start with capital letter, one space after colon.
  - `Fixes` format: `Fixes: <12-char-SHA1>(<original-commit-title>)`.

### 2. Mandatory Requirements
- **All commits must be signed**: Use `git commit -s` to auto-add `Signed-off-by` to footer. Ensure this line is last.
- **Remote repositories**:
  - `origin`: Personal fork (for pushing code).
  - `upstream`: Main project repo (for submitting Pull Requests).

### 3. Verification Gate for Dependency / Setup Changes

- If the staged changes modify a ROS package `package.xml` dependency declaration (`depend`, `exec_depend`, `build_depend`, `test_depend`, etc.), or global setup/build workflow files such as `scripts/setup.sh`, `scripts/build.sh`, `scripts/setup/platforms/*.sh`, `scripts/setup/verify_env.sh`, `scripts/install_ros.sh`, top-level `CMakeLists.txt`, or top-level `pyproject.toml`, the eventual PR description must include real Ubuntu 22.04 and openEuler Embedded Docker `setup.sh + build.sh` verification results.
- ROS package-local `setup.py` changes do **not** by themselves trigger this dual-platform setup/build gate. Examples that do not trigger it alone: console entry points, Python package metadata, or Python-only `install_requires` edits.
- When the gate is triggered during commit preparation, explicitly remind the user before committing/pushing that dual-platform verification is required for the PR. If the user asks or authorizes the agent to run it, call `ibrobot-docker-verify` and `ibrobot-docker-verify-oee` before updating the PR description. If the user does not authorize running verification, do not fabricate results; record that verification is still required for PR submission.

## Execution Steps

### Status Determination
If user explicitly requests **local commit only** (e.g., "commit to local", "only commit no push"), execute **only Phase 1, 2, and local commit part of Phase 3**. Skip push to remote, PR link, and PR description generation.

### Phase 1: Check and Summarize
1. Run `git status` in root directory and `libs/lerobot` separately.
2. Summarize pending changes to user.

### Phase 2: Compose Commit Message
1. Help user draft commit message (Title, Body, Footer) following specifications above.
2. **Validate**: Check title length, format, blank lines, and Chinese characters.
3. **Confirm staging scope**: Show the user which files will be committed, explicitly noting any excluded files (especially `libs/lerobot`).
4. **Check verification gate**: Inspect the staged file list for ROS package `package.xml` dependency changes or global setup/build workflow changes. If triggered, remind the user that Ubuntu + openEuler dual-platform Docker verification is required before the PR can be considered ready.

### Phase 3: Execute Commit and Push

**Important**: `libs/lerobot` is patch-managed and must NOT be committed to the root repo during normal commits. Only commit the files related to the current task.

For root repository:
1. Stage only the relevant files: `git add <specific-paths>` (NOT `git add .`).
2. If `libs/lerobot` was accidentally staged, unstage it: `git restore --staged libs/lerobot`.
3. Verify staging area with `git diff --cached --stat` before committing.
4. **Lint staged Python files only**: Run ruff check and format **only on the files you are about to commit**, NOT on the entire project:
   ```bash
   # Get the list of staged Python files
   STAGED_PY=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')
   if [ -n "$STAGED_PY" ]; then
     ruff check $STAGED_PY && ruff format $STAGED_PY
   fi
   ```
   If ruff auto-fixed or reformatted any file, re-stage it: `git add <fixed-files>`.
   **Never** run bare `ruff check --fix .` or `ruff format .` — that modifies unrelated files and pollutes the commit.
5. Run `git commit -s`.
6. If NOT "local commit only":
   - Execute `git push origin <branch>` (use `--force-with-lease` for amend push).
   - **Get remote info**: Extract username and repo name via `git remote get-url origin`.
   - **Check for existing PR**: After push, check whether this branch already has an open PR on the target repo. If the push output or remote hook response contains a PR/MR URL, extract the PR number. Otherwise, query via `pr_management.py --fetch-info` or check AtomGit UI.
   - **If an existing PR is found**: The PR description is now stale. You **must** synchronize it:
     1. Run `python3 pr_management.py --pr <NUM> --fetch-info` to get full PR context (all commits + diff).
     2. Analyze all commits in the PR and regenerate a complete PR description (Chinese by default) covering all changes.
     3. If the PR context triggers the verification gate, include the user's provided dual-platform verification results, or run `ibrobot-docker-verify` and `ibrobot-docker-verify-oee` only after explicit user authorization.
     4. Write the updated `description.json` and run `python3 pr_management.py --pr <NUM> --update-pr description.json`.
   - **If no existing PR**: Generate AtomGit PR link: `https://atomgit.com/<username>/IB_Robot/merge_requests/new?source_branch=<current-branch>` and compose PR description from commit message body.

For submodule `libs/lerobot` (only when user explicitly asks):
1. Change to directory -> `git add` -> `git commit -s`.
2. If NOT "local commit only":
   - Execute `git push origin <branch>`.
   - Record commit hash.
3. **Remind the user**: Changes in `libs/lerobot` should normally be exported as patches via `ibrobot-lerobot-patch` skill instead of being committed directly to the root repo.

## Common Commands Reference
- Push to personal fork: `git push origin <branch>`
- Force push (amend): `git push origin <branch> --force-with-lease`
- Signed commit: `git commit -s` (opens editor or use -m flag)
- Undo last commit (keep changes): `git reset --soft HEAD~1`
