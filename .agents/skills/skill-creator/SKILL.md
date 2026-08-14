---
name: skill-creator
description: "Create or refactor IB-Robot Agent skills following the agentskills.io specification. Use when user asks to 'create a new skill', 'add a skill', 'refactor a skill', 'write a SKILL.md', '新建 skill', '创建技能', '重构 skill', '编写 SKILL.md', or needs guidance on skill structure, YAML frontmatter, progressive disclosure, or reference file organization. Enforces the Agent Skills open standard (name/description constraints, < 500 line main doc, references/ for detailed content)."
---

# Skill Creator

Create or refactor IB-Robot Agent skills that comply with the [Agent Skills open standard](https://agentskills.io/specification).

## When to Use

- User asks to create a new skill under `.agents/skills/`
- User asks to refactor an existing skill to follow progressive disclosure
- User asks about SKILL.md format, frontmatter fields, or reference file organization
- User asks to validate a skill against the specification

## Agent Skills Open Standard (Summary)

Full spec: https://agentskills.io/specification

### Required Directory Structure

```
<skill-name>/
├── SKILL.md          # Required: YAML frontmatter + Markdown body
├── references/       # Optional: detailed docs loaded on demand
└── scripts/          # Optional: executable code
```

### Frontmatter Fields

| Field | Required | Constraints |
|-------|----------|-------------|
| `name` | Yes | 1-64 chars; lowercase `a-z`, `0-9`, hyphens only; must not start/end with hyphen; no consecutive hyphens; **must match parent directory name** |
| `description` | Yes | 1-1024 chars; must describe **what** the skill does **and when** to use it; include specific trigger keywords (中英双语) |
| `license` | No | License name or reference to bundled license file |
| `compatibility` | No | 1-500 chars; environment requirements (product, system packages, network) |
| `metadata` | No | Map of string → string for additional properties |
| `allowed-tools` | No | Space-separated pre-approved tools (experimental) |

### Body Constraints

- **Keep main `SKILL.md` under 500 lines** (recommended < 5000 tokens)
- Move detailed reference material to separate files in `references/`
- Use relative paths from skill root for file references
- Keep file references **one level deep** from `SKILL.md`

### Description Best Practices

The `description` is the **most important field** — agents use it to decide whether to activate the skill.

Must include the **Trigger Triad**:
1. **Capability** (what it does)
2. **Conditions** (when to use it)
3. **User vocabulary** (how users search for it)

Example:
```yaml
description: "Convert ACT or SmolVLA models to RKNN deployments for RK3588. Use when users mention 'rknn', 'RK3588', 'rknn-toolkit2', 'convert to rknn', 'RKNN deployment', '模型转换', 'rknn转换', or 'NPU 推理'."
```

## Progressive Disclosure (4 Layers)

Skills are loaded progressively. Structure content to take advantage of this:

| Layer | What loads | Token cost | When |
|-------|-----------|------------|------|
| 0. Metadata | `name` + `description` | ~100 tokens | Startup (all skills) |
| 1. Main SKILL.md | Full body | < 5000 tokens | Skill activated |
| 2. references/ | Specific reference file | As needed | Referenced in instructions |
| 3. scripts/ | Executed, not loaded | Variable | Invoked by agent |

### Main SKILL.md Should Contain

- Routing logic (when to use this skill vs. others)
- Core principles and hard constraints
- Workflow skeleton (steps as brief descriptions)
- Essential variables/tables
- **Internal References table** pointing to `references/` files

### references/ Should Contain

- Detailed step-by-step commands
- Troubleshooting tables
- JSON/YAML examples
- Edge cases and pitfalls
- Each file starts with `## When to Read` section

## Internal References

Read these before creating or refactoring a skill:

| Purpose | Reference |
|---------|-----------|
| agentskills.io 规范速查（frontmatter 字段、description 编写要点、渐进式披露 4 层模型）；完整规范见 https://agentskills.io/specification | `references/specification.md` |
| IB-Robot 项目特定的 skill 创建流程、索引同步规则、命名约定、PR 提交检查清单 | `references/project-conventions.md` |

Do not expose these references as separate skills.

## Workflow

### Creating a New Skill

1. **Check if the skill already exists** — search `.agents/skills/` for similar names/descriptions
2. **Read** `references/specification.md` for the full standard
3. **Read** `references/project-conventions.md` for IB-Robot specific rules
4. **Create the directory** `.agents/skills/<skill-name>/`
5. **Write `SKILL.md`** with valid frontmatter and body (< 500 lines)
6. **Create `references/`** if the body exceeds 200 lines or has distinct sub-scenarios
7. **Sync the index trio** (see below)
8. **Validate** with the checks below

### Refactoring an Existing Skill

1. **Read the current SKILL.md** and identify content that can move to `references/`
2. **Apply the split pattern**:
   - Keep routing/principles/skeleton/variables in main doc
   - Move detailed commands, examples, troubleshooting to `references/<topic>.md`
   - Add `## Internal References` table to main doc
   - Add `## When to Read` section to each reference file
3. **Target ≤ 200 lines** for the main doc (hard limit: 500 lines per spec)
4. **Sync the index trio** if the skill name or description changed

## Index Trio Synchronization (MANDATORY)

When adding, renaming, or removing a skill, update **all three** files:

1. **`AGENTS.md`** — update the 「Agent 技能索引」 section table
2. **`.agents/skills/README.md`** — update the 技能清单 table and category description
3. **`.agents/skills/intro/SKILL.md`** — update the 技能分类列表 table and 使用示例 section

## Validation Checklist

Before submitting a new or refactored skill, verify:

- [ ] `name` matches the directory name exactly
- [ ] `name` is 1-64 chars, lowercase + numbers + hyphens, no leading/trailing/consecutive hyphens
- [ ] `description` is 1-1024 chars and includes what + when + trigger keywords
- [ ] Main `SKILL.md` body is under 500 lines
- [ ] Detailed content moved to `references/` with `## When to Read` sections
- [ ] File references use relative paths, one level deep
- [ ] Index trio (AGENTS.md, README.md, intro/SKILL.md) is synchronized
- [ ] No `assets/` directory unless containing actual static resources

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Description too brief (only "what", no "when") | Add trigger conditions and user vocabulary |
| Main doc > 500 lines with no references/ | Split into references/ files |
| Reference files without "When to Read" | Add `## When to Read` section at the top |
| Forgetting to sync index trio | Update AGENTS.md + README.md + intro/SKILL.md |
| Using `assets/` for non-static files | Use `scripts/` for executables, `references/` for docs |
| Deeply nested reference chains | Keep references one level deep from SKILL.md |
| Description with unescaped quotes | Use YAML `>-` block scalar or escape properly |

## When To Use This Skill

Use for:

- Creating new IB-Robot Agent skills
- Refactoring existing skills to follow progressive disclosure
- Validating skills against the agentskills.io specification
- Answering questions about SKILL.md format or skill structure

Do not use for:

- Modifying skill content that doesn't affect structure (just edit the file directly)
- Creating non-skill documentation (use regular Markdown files)
