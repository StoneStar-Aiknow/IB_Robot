# usb_cam 补丁构建

## 来源

- 官方来源为 [ros-drivers/usb_cam](https://github.com/ros-drivers/usb_cam)，setup 使用
  [AtomGit 镜像](https://atomgit.com/ib_robot/usb_cam)，tag `0.8.1`，固定提交
  `cb2ae6bc0a312de629f84fef933fb3e07dac1ef8`。主仓库只保存 submodule 指针，不 vendoring 源码。
- 唯一补丁为上游 [PR #378](https://github.com/ros-drivers/usb_cam/pull/378)，提交
  `c3c230c6d4c607226502d83e4586cc0844afc18e`。保留 DipFlip 的作者、日期和提交说明；只适配
  0.8.1 的源码路径及 100 列 lint 所需注释换行。版权和上游 `LICENSE` 文件保持不变。
- PR #378 修复 `raw_mjpeg` 的分支判断，不处理 JPEG 实际载荷长度；本栈不含本地驱动扩展补丁。

## 使用

`full` profile 下载 `src/usb_cam` 并应用补丁；`inference` 和 `training` 不处理此相机驱动。
补丁应用复用 `scripts/setup/ros_third_party.sh` 已有的 commit 检查与 `_apply_ros_patch_once`
共享函数，单补丁直接登记，不额外引入应用脚本或 series 清单。

```bash
source .shrc_local
./scripts/setup.sh --profile full --only-patch --yes
./scripts/build.sh --mixin test -- --packages-select usb_cam
colcon test --merge-install --packages-select usb_cam
colcon test-result --verbose
ros2 pkg prefix usb_cam
```

最后一条应指向当前工作区 `install`。`--only-patch` 会准备 profile 选择的源码和补丁，可能联网，
但跳过依赖安装和编译；编译由 `build.sh` 完成。子模块 HEAD 保持固定上游提交，补丁保留为工作区修改。
新增补丁时用 `git format-patch` 导出，并像现有 ROS 第三方补丁一样在共享 dispatcher 显式登记。
`bash scripts/setup/tests/test_ros_patch_helpers.sh` 覆盖共享 apply helper 和 usb_cam 接线。
