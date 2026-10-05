# 生成文档并检查输出

## When to Read

- 执行第 5 步「生成中文文档并检查输出」时（开始生成前必须完整阅读）

## 生成文档

```bash
python -X utf8 <skill_dir>/scripts/deepwiki_processor.py --input-dir <target_dir> --output-dir <output_dir> --config-file <target_config> --branch <branch>
```

`<skill_dir>` 指本技能目录（`scripts/`、`references/` 的父目录），执行时替换为实际路径。命令写出页面与目录/主 `index.rst`，并把链接转换为输出相对路径或 AtomGit URL；**输出目录会被重建（先清空）**。

`--source-config-file <source_config>`（可选）：源语言配置，作为链接解析的标题别名表，用于解析译文中残留的英文标题链接。若源配置与 `target_config` 同目录且名为 `doc_config.json`，可省略（脚本自动发现）；否则建议显式传入。

不要手工修改生成结果；应修复 `target_config` 或 `raw_md_zh` 后重新生成。

## 验证清单

生成后检查：

1. 每个配置标题都作为且只作为一个译文 Markdown 文件的第一个 H1 出现。
2. 每个译文 Markdown 文件都被 `hierarchy` 使用。
3. `deepwiki_processor.py` 不输出 `Configured title missing from input` warning。
4. `deepwiki_processor.py` 不输出 `Input page not used by hierarchy` warning。
5. 围栏代码块数量与英文源文件一致。
6. 在处理器转换前，Markdown 链接目标和图片目标与英文源文件一致。
7. `link_conversions.json` 中的转换符合预期，没有被翻译过的路径或 URL。
8. 抽查链接显示文字：URL 与锚点不变，仅显示文本可翻译（对照英文源判断）。
9. 生成的 `index.rst` toctree 条目指向实际存在的生成文件。

`(dangling anchor)` 条目检查归第 6 步（见 `references/link-repair.md`），本步不重复检查。

## 失败处置

先完整执行上方验证清单，汇总全部失败项，再按下表统一修复源文件；全部修复完成后重新生成并完整重检一次，禁止每修一项就重新生成一次。

- `Configured title missing from input`（译文 H1 与配置标题不一致）：对照第 2 步构建的「文件名 -> 中文 H1」映射判断哪边错——译文 H1 写错则改 `raw_md_zh` 的 H1，配置标题译错则改 `target_config` 的标题。
- `Input page not used by hierarchy`（存在未被配置使用的译文页面）：核对 `target_config.hierarchy` 与该文件名，三选一——补充 hierarchy 条目 / 删除多余译文文件（须经用户确认）/ 纠正文件名。
- 保真失败（围栏代码块数量不符、路径或 URL 被翻译、链接显示文字的 URL 变了）：修 `raw_md_zh` 对应内容。
- 已知坑——UTF-8 BOM：带 BOM 的源文件会让首行 H1 解析失败（表现为 `Configured title missing from input`）。读写统一用 `utf-8-sig`（`verify_anchors.py` 已内置；临时排查脚本也须如此）。
