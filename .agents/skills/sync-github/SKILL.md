---
name: sync-github
description: "Sync the IB_Robot master branch from the AtomGit origin fork to the GitHub mirror. Use when user asks to 'sync to github', 'push to github', 'mirror to github', '同步到github', '推送github', '备份到github', or 'sync-github'."
---

# 同步到 GitHub (Sync to GitHub)

将 `IB_Robot` 仓库的 `origin` 远端（个人 AtomGit fork）的 `master` 分支同步推送到 GitHub 镜像仓库。

## 执行步骤

当用户请求同步到 GitHub 时，按以下步骤执行。

### 1. 确认 GitHub remote 配置（不自动覆盖已有配置）

检查当前 Git 配置中是否已经存在名为 `github` 的远端：

```bash
git remote -v
```

按以下策略处理：

| 情况 | 操作 |
|------|------|
| `github` remote **不存在** | 询问用户 GitHub 仓库地址，确认后 `git remote add github <url>` |
| `github` remote **已存在** | **使用其当前 URL，不自动覆盖**。直接进入下一步 |

**禁止**：不询问用户就 `git remote set-url github <硬编码地址>`。每个贡献者的 GitHub fork 地址不同，自动覆盖会破坏其他人的配置。

如果用户明确要求更新 remote 地址，才执行 `git remote set-url github <new_url>`。

### 2. 拉取 origin 的最新 master

```bash
git fetch origin master
```

### 3. 推送到 GitHub

将 `origin/master` 分支直接推送到 `github` 远端的 `master` 分支（点对点推送，不切换本地工作分支）：

```bash
git push github origin/master:refs/heads/master
```

## 常用命令组合

```bash
git fetch origin master && git push github origin/master:refs/heads/master
```

## 何时使用

- 用户要求把 AtomGit fork 的 master 同步/镜像到 GitHub
- 用户说"同步到 github"、"push 到 github"、"备份到 github"

不用于：
- 向 GitHub 提交 PR（GitHub 仅为镜像，PR 流程走 AtomGit）
- 普通 git push 到 origin（origin 是 AtomGit fork）
