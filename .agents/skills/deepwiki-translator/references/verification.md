# 翻译输出验证

## When to Read

- 执行第 6 步（验证输出）时
- 处理 `deepwiki_processor.py` 的 warning 时
- 检查链接转换报告时

## 第 5 步：生成中文文档

运行 `deepwiki_processor.py` 时使用：

```bash
python deepwiki_processor.py --input-dir <target_dir> --output-dir <output_dir> --config-file <target_config> --branch <branch>
```

不要手工修改生成结果；应修复 `target_config` 或 `raw_md_zh` 后重新生成。

## 第 6 步：验证输出

验证是必需步骤：

1. 每个配置标题都作为且只作为一个译文 Markdown 文件的第一个 H1 出现。
2. 每个译文 Markdown 文件都被 `hierarchy` 使用。
3. `deepwiki_processor.py` 不输出 `Configured title missing from input` warning。
4. `deepwiki_processor.py` 不输出 `Input page not used by hierarchy` warning。
5. 围栏代码块数量与英文源文件一致。
6. 在处理器转换前，Markdown 链接目标和图片目标与英文源文件一致。
7. `link_conversions.xlsx` 中的转换符合预期，没有被翻译过的路径或 URL。
8. 生成的 `index.rst` toctree 条目指向实际存在的生成文件。

## 输出摘要

完成后输出：

- 使用的翻译模式（全量或增量）。
- 本地化配置路径。
- 翻译的配置标题数量。
- 翻译的 Markdown 文件数量（增量模式下同时报告跳过的文件数）。
- 缺少配置映射的源文件。
- 空标题或重复标题情况。
- 链接验证结果。
- `deepwiki_processor.py` warnings，如有。
- 生成目录和链接转换报告路径。
