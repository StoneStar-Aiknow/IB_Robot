---
name: "doc-link-validator"
description: "校验 Markdown/RST 文档树中本地相对链接（文件与锚点存在性）、外部 URL（可达性）、AtomGit 源链接（blob/PR/commit/issue，API 存在性），输出 JSON 报告。Use when users mention 'check broken links', 'link validation', 'dead link check', 'AtomGit link check', '断链检查', '死链检查', '链接体检', '链接校验', or '交付前体检'."
---

# 链接校验器

对文档树做只读链接体检：本地相对链接查目标文件与锚点存在性，外部 URL 发真实 HTTP 请求，AtomGit 链接走 `api.atomgit.com` 逐类验证存在性。唯一入口 `scripts/link_validator.py`，单脚本零内部依赖，可直接整体拷贝到其他仓库使用。

## 何时调用

- 用户要求检查文档断链、死链、链接有效性。
- 文档交付或发布前的链接体检。
- 批量验证 AtomGit 源链接（`/blob/` 文件、PR、commit、issue、milestone、用户/组织）。
- 作为 `deepwiki-translator` 等生成流程之后的可选后置校验步骤。

## 输入参数

| 参数 | 必填 | 说明 | 默认值 |
|---|---|---|---|
| `paths` | 是（位置参数，可多个） | 待扫描的文件或目录（目录递归） | — |
| `--root` | 否 | 本地链接解析根；越出此根的相对链接判 broken | `.` |
| `--report` | 否 | JSON 报告路径 | `<首个扫描目录的父目录>/reports/link_validation.json` |
| `--config` | 否 | 含 `atomgit.token` 的 JSON 配置路径，支持 `$ATOMGIT_TOKEN` 占位符展开 | `config.json` |
| `--access-token` | 否 | AtomGit 访问令牌，优先级最高 | — |
| `--timeout` | 否 | 单请求超时（秒） | `10` |
| `--max-workers` | 否 | 远程链接并发数 | `16` |
| `--fail-on-inconclusive` | 否 | 存在 inconclusive 时也返回非零退出码 | 关闭 |

## 执行流程

1. 确认扫描目标与 `--root`（本地链接存在性以 root 为界，通常指向被检文档的站点根或生成目录本身）。
2. 运行校验（`<skill_dir>` 指本技能目录，执行时替换为实际路径；`--root` 等相对路径参数以当前工作目录为基准）：

    ```bash
    python -X utf8 <skill_dir>/scripts/link_validator.py <paths...> \
        --root <root> \
        --report <report.json> \
        --config config.json
    ```

3. 命令产出：控制台只打印非 valid 项（格式 `STATUS 文件:行号 URL (detail)`）；JSON 报告含 `checked_summary`（全量各状态计数）、`reported_summary`（非 valid 计数）与非 valid 明细。命令输出与退出码可直接驱动后续动作；报告文件用于长输出兜底、修复前后对比与流程集成。

## 结果判定

| 状态 | 含义 | 处置 |
|---|---|---|
| `valid` | 目标存在且可访问 | 无需处理 |
| `broken` | 目标不存在或不可达：本地文件缺失、链接越出 `--root`、空链接目标、外部 404/5xx、AtomGit 仓库/PR/commit/issue 不存在、`/blob/` 与 `/tree/` 路由词与目标类型不匹配、`#L10-L20` 行号超出文件实际行数、`/bolb/` 拼写错误 | 必须修复源文件 |
| `auth_error` | 外部或 AtomGit 返回 401/403 | 确认访问权限或 token 是否有效 |
| `error` | 网络层失败或超时 | 排查网络后重跑，不计为文档缺陷 |
| `inconclusive` | 无法确认：无 token 的 AtomGit 链接、fragment 近似算法未命中 | 按发布要求决定是否人工复核；需严格阻断时加 `--fail-on-inconclusive` |

退出码：`0` 通过（允许存在 inconclusive）；`1` 存在 `broken` / `auth_error` / `error`（或开启 `--fail-on-inconclusive` 时的 inconclusive）；`2` token 配置错误。

**`broken` 与 `inconclusive` 必须区分对待**：前者是确定缺陷，后者是"没法确认"——把 inconclusive 当 broken 修或直接忽略，都是本技能最常见的误用方式。

## Token 配置（AtomGit）

解析优先级：`--access-token` > `--config` 指定 JSON 的 `atomgit.token` 字段（值支持 `$VAR` / `${VAR}` 环境变量展开）> 环境变量 `$ATOMGIT_TOKEN`。

无 token 时 AtomGit 链接一律报 `inconclusive` 而非 `broken`，不会误报。受限仓库需要真实校验时，确保 `config.json` 中 `atomgit.token` 可通过 `$ATOMGIT_TOKEN` 展开，或显式传入 `--access-token`。

## 行为边界

校验前清洗以下内容（其中链接不参与校验）：

- ` ``` ` / ` ~~~ ` 围栏代码块（含 ` ```mermaid `）。
- RST `.. mermaid::` 指令体（按缩进界定指令边界）。

以下内容**会被校验**，属于有意取舍：

- 行内代码 `` `...` `` 中的链接与裸 URL——渲染语义上不是链接，但宁可多报不漏报；报告定位容易，人工识别后忽略即可。
- 单行 Markdown 链接——跨行链接不检测（匹配文档导出格式）。
- `.. note::` 等非 mermaid 的 RST 指令体。

锚点（fragment）校验是近似算法：支持 `{#explicit}` 显式锚点、CJK 标题 slug、重复标题 `-1` 后缀、HTML `id` 属性；与实际渲染器（GitHub/Sphinx）可能存在差异，因此未命中只报 `inconclusive`。

其他：外部请求 HEAD 优先，遇 405/501/403/网络失败降级 GET；非 ASCII URL（如中文路径）自动百分号编码后请求；文件读取 utf-8 失败回退 gbk；同一 URL 多处引用只发一次请求。

## 约束

- 本技能只读：不修改任何被扫描文档，唯一写出物是 JSON 报告。
- 不用于校验任意 URL 清单——输入必须是文档文件或文档目录。
