# 链接与锚点修复

## When to Read

- 第 6 步「锚点与链接校验」退出码非 0，或 `link_conversions.json` 出现 `(dangling anchor)` 条目时
- 需要复用标准修复循环时（见「链接修复循环」）
- 需要了解三层防护机制或源链接形态规范时

## 扫描的四类问题

`verify_anchors.py` 扫描生成目录下所有 Markdown 文件，检查：

1. 数字页面 ID 锚点（如 `#16.1`）不应在生成结果中存活——它们必须被处理器转换为相对链接。
2. 页内锚点（`#section`）必须能匹配本文件某个标题的 GitHub 风格 slug。
3. 相对链接（`./x.md`、`../y/x.md`）的目标文件必须存在。
4. 带锚点后缀的相对 Markdown 链接的锚点必须能匹配目标文件的标题 slug。

围栏代码块内容被忽略；外部 `http(s)` 链接不做检查。退出码 0 表示干净，非 0 表示存在问题；存在问题时应修复 `target_config` 或 `raw_md_zh` 后重新生成并重跑本步骤，禁止手工修补生成结果。

`link_conversions.json` 中的 `(dangling anchor)` 条目（第 2 层防护的记录）与第 2 类问题同源：译文中的页内锚点在本文件无对应标题，通常是译文改写了锚点。

历史教训：翻译阶段改写 `(#16.1)` 这类数字锚点后，处理器会把它当作普通页内锚点静默透传，导致生成结果中留下永远无法命中的悬空锚点。第 2 层在生成时拦截，本步骤在生成后兜底，两层缺一不可。

## 三层防护一览

| 层级 | 位置 | 时机 | 作用 |
|------|------|------|------|
| 第 1 层 | `deepwiki_processor.py` 链接转换 | 生成时 | 数字页 ID 锚点（`#16.1`）转换为输出相对链接 |
| 第 2 层 | `deepwiki_links.py` 内联校验 | 生成时 | 非数字页内锚点对本地标题 slug 匹配，不匹配则 warning + `(dangling anchor)` 记录 |
| 第 3 层 | `verify_anchors.py` | 生成后 | 全量扫描四类问题：数字锚点残留、页内锚点悬空、相对链接失效、跨文件锚点错误 |

第 2、3 层配合工作：第 2 层在生成日志中即时暴露问题链接所在文件，第 3 层保证交付前全绿（退出码 0）。

## 链接修复循环（复用流程）

发现链接或锚点问题时的标准处理顺序。核心原则：**只改源（`raw_md_zh` 或 `target_config`），改完重新生成，禁止手工修补生成结果**。

循环内命令统一使用占位符：`<skill_dir>` 定义于 SKILL.md 执行流程开头，`<target_dir>`、`<output_dir>` 等对应 SKILL.md「输入参数」表中的同名参数，执行时替换为实际值；维护本文件时保持占位符形式，禁止写入硬编码路径。

1. **定位**：从 `verify_anchors.py` 输出或 `link_conversions.json` 的 `(dangling anchor)` 条目找到问题链接所在文件与行号。
2. **诊断源文件**：打开 `raw_md_zh` 中对应源文件，确认链接属于哪类问题（见下方源链接形态规范）。
3. **修源**：在 `raw_md_zh` 中改链接；若问题源于标题翻译改写了锚点，则恢复锚点或对应标题。
4. **重新生成**（初生成时若显式传入过 `--source-config-file`，重跑同样传入）：

   ```bash
   python -X utf8 <skill_dir>/scripts/deepwiki_processor.py \
       --input-dir <target_dir> \
       --output-dir <output_dir> \
       --config-file <target_config> \
       --branch <branch>
   ```

5. **全量校验**：

   ```bash
   python -X utf8 <skill_dir>/scripts/verify_anchors.py \
       --output-dir <output_dir>
   ```

   退出码非 0 时回到第 1 步。
6. **比对**（可选，确认重新生成未引入意外差异）：`git diff --no-index --ignore-cr-at-eol <旧输出> <新输出>`。
7. **保真复检**：链接批量修改后，按 `references/output-pipeline.md`「验证清单」中的保真项复检（围栏代码块数量、`link_conversions.json` Converted 字段无路径或 URL 被翻译、抽查链接显示文字 URL 不变），防止批量修改引入新问题。

### 源链接形态规范（raw_md_zh 中）

| 链接目标 | 正确形态 | 示例 |
|----------|----------|------|
| 同目录 wiki 页面（输出后仍在同一目录） | `./x.md` | `](./control_mode_architecture.md)` |
| 跨目录 wiki 页面 | `../dir/x.md`（从子目录页指向根级页）或对应输出相对路径 | `](../architecture.md)` |
| 仓库文件（非 wiki 页面） | 裸仓库相对路径，不加 `./` 前缀——处理器会转换为 AtomGit 绝对 URL | `](src/robot_config/README.md)`、`](AGENTS.md)` |

判别方法：目标文件的 H1 出现在 `target_config` 的 `hierarchy` 中即为 wiki 页面；否则是仓库文件链接，保持裸路径。

### 已知坑

- **代码块误伤**：批量改写链接时必须先保护围栏代码块（` ``` ` 分隔），否则示例中的链接文本会被改坏。
- **`../` 可疑项**：根级页面（如 `system_architecture.md` → `architecture.md`）映射后位于输出根目录，子目录页指向它用 `../architecture.md` 是正确形态，不要"修复"成 `./architecture.md`。
- **手工修补陷阱**：直接编辑生成结果会造成与重新生成输出不一致——本次修复的官方输出中就残留过 3 处手工修补与 regen 结果不一致的行。
