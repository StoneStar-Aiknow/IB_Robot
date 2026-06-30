# IB-Robot DeepWiki 文档迁移与转换指南

本目录用于承接 DeepWiki 输出的 IB-Robot 文档，并通过 Agent 技能完成拆分、配置、翻译、生成与校验。用户直接依次使用 `deepwiki-config`、`deepwiki-translator` 和 `mermaid-syntax-validation` 这 3 个技能，即可完成文档迁移与转换。

## 最新流程概览

推荐按以下顺序执行：

1. `deepwiki-config`: 基于 DeepWiki 目录结构生成或更新 `doc_config.json`。
2. `deepwiki-translator`: 拆分 DeepWiki Markdown、执行 config-first 翻译、运行 `deepwiki_processor.py` 生成文档，并执行链接检查。
3. `mermaid-syntax-validation`: 对生成后的文档执行 Mermaid 语法检查、最小修复和渲染验证。

下文保留每个阶段涉及的输入、产出和脚本命令，便于人工复核或排查问题。

### 1. 提取 DeepWiki 内容

使用 DeepWiki MCP 获取远程仓库文档内容。

- **内容工具**: `mcp_deepwiki_read_wiki_contents`
- **结构工具**: `mcp_deepwiki_read_wiki_structure`
- **产出**: `docs/migration/IB_Robot_doc_raw.md`
- **说明**: 原始文件包含所有页面内容，页面通常以 `# Page: <Title>` 分隔。

`read_wiki_contents` 用于获得待拆分的 Markdown 原文；`read_wiki_structure` 用于后续配置生成。

### 2. 生成配置

使用 `deepwiki-config` 技能根据 DeepWiki 目录结构自动生成 `doc_config.json`。

- **技能**: `deepwiki-config`
- **输入**: DeepWiki `read_wiki_structure` 返回的页面层级，或由技能直接读取目标仓库结构
- **产出**: `docs/migration/doc_config.json`
- **配置内容**:
  - `id_to_label`: DeepWiki 页码到稳定标签的映射。
  - `title_to_label`: 页面标题到稳定标签的映射。
  - `hierarchy`: 生成目录、页面文件和子页面归属关系。

配置是后续翻译和生成的单一事实来源。更新已有配置时，应优先保留既有 label，只为新增页面生成新 label，避免破坏历史链接。

### 3. 拆分 Markdown

使用 `deepwiki-translator` 技能脚本中的 `split_md.py` 将 DeepWiki 原始大文件拆成按页面组织的 Markdown 文件。

- **脚本**: `.agents/skills/deepwiki-translator/scripts/split_md.py`
- **命令**:

```bash
python .agents/skills/deepwiki-translator/scripts/split_md.py \
    docs/migration/IB_Robot_doc_raw.md \
    docs/migration/raw_md
```

- **产出**: `docs/migration/raw_md/*.md`

拆分后的文件名应保持稳定，后续翻译也应沿用相同文件名。

### 4. 翻译与生成

使用 `deepwiki-translator` 技能完成配置本地化、Markdown 翻译和文档生成。

该技能采用 config-first 流程：先本地化 `doc_config.json` 的标题，再翻译页面，并强制每个页面的 H1 与本地化配置保持一致。

#### 4.1 本地化配置

- **源配置**: `docs/migration/doc_config.json`
- **目标配置**: `docs/migration/doc_config_zh.json`
- **必须保持不变**:
  - `id_to_label` 的 key/value。
  - `title_to_label` 的 label 值。
  - `hierarchy` 的目录名、文件名和 `subs` key。
- **只翻译**:
  - `title_to_label` 的标题 key。
  - `hierarchy.*.title`。
  - `hierarchy.*.subs.*` 标题值。

目录型页面的概述页标题应按技能要求加上“概述”后缀，避免和章节名冲突。

#### 4.2 翻译 Markdown

- **源目录**: `docs/migration/raw_md`
- **目标目录**: `docs/migration/raw_md_zh`
- **模式**: `full` 或 `incremental`

翻译时需要保持：文件名、链接目标、图片路径、锚点、代码块、Mermaid 图、HTML/RST 指令、命令、API、topic、参数名等不变。每个翻译后文件的第一个 H1 必须精确等于 `doc_config_zh.json` 中对应页面标题。

#### 4.3 生成文档

翻译完成后，使用技能脚本中的 `deepwiki_processor.py` 生成 Sphinx 可消费的 Markdown 文档结构。

- **脚本**: `.agents/skills/deepwiki-translator/scripts/deepwiki_processor.py`
- **命令**:

```bash
python .agents/skills/deepwiki-translator/scripts/deepwiki_processor.py \
    --input-dir docs/migration/raw_md_zh \
    --output-dir docs/migration/ib_robot_zh \
    --config-file docs/migration/doc_config_zh.json \
    --source-config-file docs/migration/doc_config.json \
    --branch master
```

- **产出**:
  - 层级化 Markdown 文档目录。
  - `link_conversions.xlsx` 链接转换报告。
  - 生成阶段的 warning 汇总。

不要手工修改生成目录；如需修复内容，应修改 `doc_config_zh.json` 或 `raw_md_zh/*.md` 后重新生成。

## 链接检查

生成完成后，使用 `deepwiki-translator` 技能脚本中的 `link_validator.py` 检查本地链接、外部链接和 AtomGit 链接。

- **脚本**: `.agents/skills/deepwiki-translator/scripts/link_validator.py`
- **命令**:

```bash
python .agents/skills/deepwiki-translator/scripts/link_validator.py \
    docs/migration/ib_robot_zh \
    --root docs/migration/ib_robot_zh \
    --report docs/migration/reports/link_validation.json \
    --config config.json
```

如需要校验 AtomGit 私有或受限接口，确保 `config.json` 中的 `atomgit.token` 可通过 `$ATOMGIT_TOKEN` 展开，或显式传入 `--access-token`。

检查结果应重点关注：

- `broken`: 必须修复。
- `auth_error`: 确认是否缺少访问权限或 token。
- `inconclusive`: 根据网络环境和发布要求决定是否复核；如需严格阻断，可增加 `--fail-on-inconclusive`。

## Mermaid 语法检查与优化

翻译和生成完成后，使用 `mermaid-syntax-validation` 技能对 Mermaid 图进行语法检查、最小修复和渲染验证。

- **技能**: `mermaid-syntax-validation`
- **检查范围**: 翻译后的 Markdown 源目录和生成后的文档目录。
- **处理原则**: 只修 Mermaid 语法，不改变图含义、节点文本、边含义、顺序或周边正文。

推荐检查内容：

1. 统计 Markdown 中的 Mermaid fence 数量。
2. 静态扫描高风险语法，例如匿名节点、旧式边标签、带括号或冒号的未加引号标签。
3. 修复后执行真实 Sphinx 构建。
4. 浏览器打开生成 HTML，确认 Mermaid 容器渲染为正常 SVG，且没有 `Syntax error in text`、`Parse error`、`Diagram error` 或 `mermaid version`。

## 增量更新建议

当 DeepWiki 源内容更新时，推荐采用增量流程：

1. 重新提取 `IB_Robot_doc_raw.md` 和 wiki structure。
2. 用 `deepwiki-config` 更新 `doc_config.json`，保留既有 label。
3. 重新运行 `split_md.py` 生成新的 `raw_md/`。
4. 用 `deepwiki-translator` 的 incremental 模式只翻译新增或明确变更的页面。
5. 重新运行 `deepwiki_processor.py`。
6. 运行 `link_validator.py`。
7. 运行 `mermaid-syntax-validation`。
