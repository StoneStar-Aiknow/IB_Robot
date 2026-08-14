# Agent Skills Open Standard — Quick Reference

> **权威来源**: https://agentskills.io/specification
>
> 本文件不复制官方规范正文（避免随上游演进漂移）。仅记录创建 skill 时最常用的速查项。
> 需要完整规范时请直接访问上述 URL。

## When to Read

- 创建新 skill 前需要快速确认 frontmatter 必填字段和约束
- 编写 `description` 时需要速查长度限制和风格要求
- 验证已有 skill 的结构是否符合规范

## Frontmatter 必填字段

| 字段 | 必填 | 约束 |
|------|------|------|
| `name` | 是 | 与目录名一致；小写字母 + 连字符；无连续连字符；不以连字符开头/结尾；≤ 64 字符 |
| `description` | 是 | ≤ 1024 字符；用 if-then 条件触发风格；包含双语关键词；面向用户意图而非实现 |

## SKILL.md 结构

```
my-skill/
├── SKILL.md          # 必需: YAML frontmatter + Markdown body
├── scripts/          # 可选: 可执行代码 (Python, Bash, JS)
├── references/       # 可选: 按需加载的详细文档
└── assets/           # 可选: 模板、图片等静态资源
```

### SKILL.md Body 约束

- 行数 ≤ 500 行；超过时应使用 `references/` 做渐进式披露
- `references/` 中每个文件应注明 "When to Read" 场景
- 主文档应包含路由表（Internal References table）指向 references

## Description 编写要点

- 用 "Use when user asks to ..." 祈使句式
- 包含触发短语（中英双语）
- 面向用户意图，不描述内部实现机制
- 在涉及多 skill 边界时用 negative scoping 厘清（"For X, use skill-Y instead"）
- 保持精炼，不重复关键词

## 渐进式披露 4 层模型

| 层 | 内容 | 加载时机 |
|----|------|----------|
| Layer 0 | `description`（frontmatter） | 发现阶段，只读 name + description |
| Layer 1 | `SKILL.md` 正文 | 被触发后加载 |
| Layer 2 | `references/*.md` | 主文档路由表指向时按需加载 |
| Layer 3 | `scripts/` 代码 | 执行具体操作时加载 |

## IB-Robot 项目特有差异

IB-Robot 项目在官方规范基础上的增量约定（命名前缀、双语关键词、索引三件套同步等）见 `project-conventions.md`，不在此重复。

## 参考链接

- 官方规范: https://agentskills.io/specification
- 项目约定: `project-conventions.md`（本目录）
- skill-creator 主文档: `../SKILL.md`
