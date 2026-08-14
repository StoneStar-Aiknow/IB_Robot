# IB_Robot 专项审查要求

## When to Read

- 在 IB_Robot 仓库做 PR review 时
- 提取 PR 上下文后，看到 `.pr.mandatory_review_checks` 含 `lerobot_gitlink_changed` 时
- 判断 PR 是否触发双平台 Docker Verification 门禁时

## 1. `libs/lerobot` gitlink 强制检查（阻塞性）

- **每个 IB_Robot PR 都必须检查 `libs/lerobot` 是否发生 gitlink / submodule
  指针变化**，不能只查看 `libs/lerobot` 目录下是否有普通文件 diff。
- 以下任一信号都表示 gitlink 发生了变化：
  - changed file 路径精确等于 `libs/lerobot`
  - 文件 mode 为 `160000`
  - diff 中出现 `Subproject commit <sha>`
- IB_Robot 默认通过 `third_party/patches/lerobot/<tag>/` 管理 LeRobot 修改。
  如果 PR 只是修改 LeRobot 源码，却直接提交了 `libs/lerobot` 新指针，而没有导出 managed
  patch，应提交 **severity=error 的阻塞性 review issue**。
- 错误 gitlink 提交可能引用仅存在于贡献者本地仓库或个人 fork 的 commit。其他开发者执行
  `git submodule update --init --recursive` 时会出现 `not our ref`、`did not contain` 或无法
  checkout，对 setup、Docker 验证和发布构建造成直接破坏。

### 合法指针升级必须同时满足

- PR 明确声明这是 LeRobot 上游 tag/base 升级，而不是普通源码修复。
- 新 commit 可以从 `.gitmodules` 声明的权威 submodule URL 获取，不能只存在于作者个人
  fork、本地 object database 或临时分支。
- `third_party/patches/lerobot/INDEX.yaml`、对应 tag 目录、`manifest.yaml` 的
  `lerobot_commit_range` 与新基线保持一致。
- 现有 patch stack 已针对新基线重放或迁移，相关 `series*.txt`、manifest 条目和过滤测试
  同步更新。
- PR 描述包含基线升级原因、patch 迁移情况以及开发者提供的 setup/build 验证。

只要上述条件有一项缺失，就不能把 gitlink 变化当作普通实现细节放行。修复建议应要求作者：

1. 将 `libs/lerobot` 指针恢复到主仓库记录的基线。
2. 在 LeRobot authoring branch 中整理源码提交。
3. 使用 `ibrobot-lerobot-patch` 工作流导出 mailbox patch。
4. 更新 `series*.txt`、`manifest.yaml` 和 `scripts/setup/tests/test_lerobot_filter.sh`。

### 审查输出要求

- 提取上下文后，先查看 `.pr.mandatory_review_checks`。出现
  `lerobot_gitlink_changed` 时必须完成人工判定，不能直接给出"无问题"。
- 对违规指针变更，issue 应定位到 `libs/lerobot`，标题明确说明"禁止直接提交 LeRobot
  submodule 指针"，并在 `fix_code` / 修复方案中给出恢复 gitlink 和导出 patch 的步骤。
- 即使 PR 同时增加了 `third_party/patches/lerobot/*.patch`，也不能自动认为 gitlink 变化
  合法；普通 patch 增量通常不应改变根仓库记录的 submodule 指针。

## 2. README / 文档联动检查（按变更内容决定）

- review 时必须判断本次提交是否改变了**用户可见**的使用方式，而不是机械地要求所有 PR 都改 README。
- 当 PR 修改了以下内容之一时，应检查对应 README / 使用文档是否需要同步更新：
  - 安装、部署、启动、构建、配置、依赖声明或运行步骤
  - 对外暴露的命令、接口、参数、launch 用法、目录约定
  - 会影响用户接入、复现、验证或排障的方法
- 如果变更只涉及内部重构、实现细节、无用户感知的代码整理，则不应为了凑要求而强行提出 README 修改意见。
- 如果判断"应该改 README / 文档但没有改"，应将其作为有效 review issue 提出。

## 3. 依赖 / setup / build 变更的 Verification 强制门禁

- 如果 PR 修改了 ROS 包的 `package.xml` 依赖声明（尤其是新增/删除/调整 `exec_depend`、`build_depend`、`depend`、`test_depend` 等），或修改了全局 setup/build 流程相关文件（如 `scripts/setup.sh`、`scripts/build.sh`、`scripts/setup/platforms/*.sh`、`scripts/setup/verify_env.sh`、`scripts/install_ros.sh`、顶层 `CMakeLists.txt`、顶层 `pyproject.toml` 等），则 **PR 描述中的 Verification 不再是可选项，而是必填项**。
- ROS 包内的 `setup.py` 普通改动（例如 console entry point、Python package metadata 或 Python-only `install_requires` 调整）不单独触发双平台 `setup.sh + build.sh` Verification 门禁；只有同一 PR 还修改了 `package.xml` 依赖声明或全局 setup/build 流程文件时才触发。
- 该 Verification 必须体现**真实执行过的验证**，并明确写出：
  - **Scenario**：在哪类干净环境中验证
  - **Method**：如何执行 setup 和 build
  - **Result**：setup / build 是否成功、是否有关键限制或失败点
- 对此类 PR，review 过程中应检查 **PR 描述中的 Verification 是否由开发者提供**，且必须同时覆盖：
  - Ubuntu 22.04 纯净 Docker 环境中的 `setup.sh + build.sh` 完整验证
  - openEuler Embedded 纯净 Docker 环境中的 `setup.sh + build.sh` 完整验证
- **review 默认只检查 PR 描述中由开发者声明的验证结果，禁止自动执行双平台 Docker 验证。**
- "review / 审查 / 帮我看看 PR"本身不等于授权执行验证。禁止审查者代替开发者运行 `ibrobot-docker-verify` 或 `ibrobot-docker-verify-oee` 来补齐 PR 描述；只有当用户在当前请求中明确要求 agent 实际执行验证（例如"你来跑一下 Ubuntu/openEuler Docker 验证""帮我实际验证 setup/build"）时，才调用对应验证 skill。
- 如果 PR 描述缺少任一平台验证说明，或只给出命令但没有结果，或验证没有覆盖 setup/build 两个阶段，都应视为**阻塞性 review 问题**，要求开发者补充。

## 4. 禁止本地重复执行 pre-commit 已覆盖的检查

- IB_Robot 的 `.pre-commit-config.yaml` 已把 `ruff --fix` 与 `ruff-format` 作为强制 pre-commit hook，且 `.git/hooks/pre-commit` 随仓库安装；开发者 `git commit` 时必然已通过 ruff 校验，**PR 上线代码不会再有 `ruff check` / `ruff format` 报错**。
- 因此 review 时**禁止**在本地做以下动作（属于重复劳动，浪费上下文且无新增信息）：
  - `git apply` / `git checkout` / `git diff` 拼接 PR diff 后跑 `ruff check` / `ruff format --check` / `pyright` / `mypy` / `py_compile` 等 lint / format / typecheck 命令
  - 切到 PR 分支跑 `colcon build` 来"验证"代码能否编译——这属于开发者侧 Verification 范畴，由 §3 门禁管理
- 替代做法：
  - **信任 PR 描述中开发者声明的 ruff / py_compile / build 结果**，除非描述缺失且 PR 触发 §2 门禁
  - 如怀疑某行有 lint / 类型 / 风格问题，**直接在 inline 评论中指出并附修复建议**，由开发者在下一次提交时让 pre-commit 自动修复，不要在本地复跑
  - 静态阅读 diff 与本仓库源码（`Read` / `Grep` / `Glob`）始终允许且推荐——这是判断架构/逻辑/命名问题的正常手段
- **例外**：用户在当前请求中明确要求"你帮我跑一下 ruff / typecheck / build 看看"时才执行相应命令；"review 这个 PR""帮我看看这个 PR"本身**不构成**授权。
