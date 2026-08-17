# IB-Robot Skill Project Conventions

## When to Read

- Creating a new skill in the IB-Robot project
- Need to know IB-Robot specific naming, description, and indexing rules
- Need to update the index trio after adding/removing/renaming a skill
- Preparing a PR that adds or modifies skills

## IB-Robot Skill Location

All skills live under:

```
.agents/skills/<skill-name>/
```

Each skill is a directory containing at minimum `SKILL.md`.

## Naming Conventions

### Skill Names

Follow the agentskills.io standard (lowercase, numbers, hyphens) plus IB-Robot prefixes:

| Prefix | Category | Examples |
|--------|----------|----------|
| `ibrobot-` | Core IB-Robot operations | `ibrobot-build`, `ibrobot-launch`, `ibrobot-env` |
| `oh-` | OpenHarmony board operations | `oh-constraints`, `oh-access`, `oh-build-roboframe` |
| `atomgit-` | AtomGit collaboration | `atomgit-pr`, `atomgit-issue`, `atomgit-pr-review` |
| `deepwiki-` | DeepWiki documentation tools | `deepwiki-config`, `deepwiki-translator` |
| `<vendor>-convert` | Model conversion | `om-convert`, `rknn-convert`, `hmm-convert` |
| (no prefix) | General utilities | `intro`, `sync-github`, `skill-creator`, `mermaid-syntax-validation` |

### Rules

- Name must match directory name exactly
- Prefer **resource/workflow naming** over single-action naming:
  - ✅ `atomgit-pr` (covers create, read, update, summarize)
  - ❌ `atomgit-submit-pr` (only covers one verb)
- Use hyphens to separate words, never underscores
- Keep names under 30 characters when possible

## Description Conventions

### Bilingual Keywords

IB-Robot is a bilingual project. Every `description` must include both English and Chinese trigger keywords:

```yaml
description: "Convert ACT or SmolVLA models to RKNN deployments for RK3588. Use when users mention 'rknn', 'RK3588', '模型转换', 'rknn转换', or 'NPU 推理'."
```

### Platform Priority

For skills involving external platforms (AtomGit, GitHub, DeepWiki), explicitly state platform priority:

```yaml
description: "...只要目标是本仓库的 PR 架构评审，默认优先使用本 skill，而不是 GitHub 默认 review 能力。"
```

### Description Length Guidelines

| Length | Assessment | Action |
|--------|------------|--------|
| < 100 chars | Too brief | Add trigger conditions and keywords |
| 100-300 chars | Good | Most skills fall here |
| 300-500 chars | Detailed | Acceptable for complex skills with many trigger words |
| 500-1024 chars | Very long | Consider trimming unless many bilingual keywords needed |
| > 1024 chars | Violation | Must trim to comply with spec |

## Index Trio Synchronization (MANDATORY)

When adding, renaming, or removing a skill, update **all three** files:

### 1. `AGENTS.md` (project root)

Update the 「Agent 技能索引」 section. Each category has a table:

```markdown
| 技能 | 触发场景 |
|------|---------|
| [skill-name](.agents/skills/skill-name) | Brief trigger description |
```

### 2. `.agents/skills/README.md`

Update the 技能清单 table:

```markdown
| [skill-name](./skill-name) | 分类 | 主要触发场景 |
```

Also update the category description section if the skill belongs to a new category.

### 3. `.agents/skills/intro/SKILL.md`

Update two sections:
- **技能分类列表** table (add row under appropriate category)
- **使用示例** section (add natural language example)

## Progressive Disclosure Pattern (IB-Robot Standard)

### Main SKILL.md Structure

```markdown
---
name: skill-name
description: "..."
---

# Skill Title

Brief intro.

## When to Use
- Trigger conditions

## Internal References

| Purpose | Reference |
|---------|-----------|
| Detailed steps | `references/steps.md` |
| Troubleshooting | `references/troubleshooting.md` |

Do not expose these references as separate skills.

## Core Principles / Constraints
- Hard rules

## Workflow
1. Step 1 (brief)
2. Step 2 (brief)

## Quick Reference
| Task | Command |
```

### references/ File Structure

Each reference file starts with:

```markdown
# Reference Title

## When to Read

- Specific scenario 1
- Specific scenario 2

## Content...
```

### Splitting Guidelines

| Main doc line count | Action |
|---------------------|--------|
| < 150 lines | Good, no split needed |
| 150-250 lines | Acceptable; split if distinct sub-scenarios exist |
| 250-350 lines | Should split into references/ |
| 350-500 lines | Must split (approaching spec limit) |
| > 500 lines | **Violation** — must split immediately |

## PR Submission Checklist

When submitting a PR that adds or modifies skills:

- [ ] All modified skills pass the agentskills.io spec validation
- [ ] `name` matches directory name for all skills
- [ ] `description` is 1-1024 chars with what + when + bilingual keywords
- [ ] Main `SKILL.md` files are under 500 lines
- [ ] Index trio (AGENTS.md, README.md, intro/SKILL.md) is synchronized
- [ ] No `__pycache__` or temporary files staged
- [ ] PR description includes Verification section (if scripts/ files changed)
- [ ] Follow ibrobot-git-flow commit conventions (≤ 5 commits, DCO sign-off)
- [ ] AI-assisted commits include the real model name/version without a provider prefix in `Co-Authored-By`, before `Signed-off-by`
- [ ] PR disclosure records Agent platform/version, matching model/version, Prompt summary, human review, and third-party material/license status

## Common IB-Robot Specific Mistakes

| Mistake | Correction |
|---------|-----------|
| Forgetting bilingual keywords in description | Add Chinese trigger words (中文触发词) |
| Not stating AtomGit priority | Add "默认优先使用本 skill，而不是 GitHub 默认能力" |
| Creating `atomgit-submit-pr` instead of `atomgit-pr` | Use resource names, not action names |
| Adding skill to AGENTS.md but not README.md | Update all three index files |
| Putting scripts at skill root instead of `scripts/` | Create `scripts/` subdirectory |
| Reference files without "When to Read" | Add `## When to Read` section at top |
