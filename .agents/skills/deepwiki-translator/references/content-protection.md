# 内容保护规则与术语表

## When to Read

- 第 4 步翻译 Markdown 页面时
- 需要判断某段内容是否可以翻译
- 需要统一术语翻译

## 内容保护规则（不得翻译或改写）

- Markdown 链接目标：`[text](target)` 中的 `target`。
- 图片目标：`![alt](target)` 中的 `target`。
- 文件路径、目录名、文件名、anchor、URL、源码行号引用。
- 空链接源码引用文本，例如 `[scripts/setup.sh:1-62]()`。
- 反引号包裹的行内代码。
- 围栏代码块内容和代码块语言标记。
- Mermaid 图全部内容：图语法、节点 ID、连线操作符、subgraph ID、展示标签（引号包裹或无引号）、节点文本及所有其他 Mermaid 标记。
- HTML 标签和属性。
- 如果出现 RST 指令和选项，也必须保持结构。
- 产品名、包名、命令、API、类名、函数名、topic、action、参数名和环境变量名。

## 可安全翻译的内容

- 段落正文。
- 非 H1 标题。
- 链接目标非空且保持不变时的链接显示文本。
- 表格中的自然语言单元格，代码/路径单元格保持原样。

## 术语表

全站保持术语一致：

| 英文 | 中文 |
|---|---|
| inference | 推理 |
| deployment | 部署 |
| teleoperation | 遥操作 |
| dataset | 数据集 |
| pipeline | 流水线 |
| action dispatch | 动作分发 |
| motion planning | 运动规划 |
| configuration | 配置 |
| validation | 验证 |
| submodule | 子模块 |
| workspace | 工作空间 |

## 默认不翻译的专有名词

除非用户明确要求，以下名称不翻译：`IB-Robot`、`LeRobot`、`ROS 2`、`MoveIt`、`openEuler`、`OpenHarmony`、`AtomGit`、`Hugging Face`、`Conda`、`venv`、`colcon`、`rosdep`。

## 链接安全分析

只要 label 和文件名保持不变，配置优先流程可以保持链接稳定。

### 高风险错误

- 翻译 `environment_setup` 这类 label 值。
- 翻译 hierarchy key 或 `subs` 文件名。
- 翻译链接目标、anchor、源码引用或仓库路径。
- 页面 H1 与本地化 `doc_config.json` 标题不一致。
- `title_to_label` 中出现重复中文标题。
- 翻译 Mermaid 图的任何部分，包括展示标签、节点文本或代码块。

### 安全做法

- 只翻译配置中的标题字符串，不翻译 label 或 key。
- 页面 H1 直接复制本地化配置中的标题。
- 可以翻译普通链接显示文本，但必须保留 target。
- 文件名和输出路径继续使用英文 slug。
