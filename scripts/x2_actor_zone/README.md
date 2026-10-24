# X2 照镜子：站位区选人 —— 启动和现场标定

站位区选人：机器人只模仿**脚站在地面区域里**的那一个人。人走出区域、或者被前面的人挡住腿，就放手回中性位。放手后 3 s 内不选新人。

区域文件不进构建。改完 stop 再 start 就生效。

本目录分两部分：

- 笔记本上用的标定工具：`field_calib.sh`、`zone_*.py`、`actor_overlay.py`。
- `duck/`：duck 上 `/root/x2-actor-run/` 的运行脚本底稿。除 `run_read.py`（新加）外，与 10-24 真机第一轮用的板上文件逐字节相同（核过 sha256），所以没有按 ruff 重排。

---

## 0. 现场速查（按顺序做）

1. 笔记本连上 duck：`ssh duck` 能登上。
2. 机器人开机、站好（厂商 STAND_DEFAULT）。**别碰机器人的头。**
3. 标定区域（第 2 节）。在笔记本上，本目录里执行：
   ```
   bash field_calib.sh
   ```
   按屏幕提示做：清场 → 一个人依次站 近左、近右、远右、远左、中间 → 点 6 下 → 看核对图 → 输入 y 上传。
4. 起整套（duck 上）：
   ```
   ssh duck
   bash /root/x2-actor-run/start.sh
   ```
5. **等 60–90 s 再开镜子**。刚起来那会儿板子满载，马上开会双臂卡跳。
6. 开镜子：双捂开（捂后脑勺相机两下，**手别碰到头**）。人站进区域，约 3 s 开始模仿。
7. 收：长捂关（捂住不放）。
8. 停整套（duck 上）：
   ```
   bash /root/x2-actor-run/stop.sh
   ```
   它会先让机器人回站立姿势，再停进程。手臂会动，周围别站人。

---

## 1. 每天起停（duck 上）

```
bash /root/x2-actor-run/start.sh              # 带区域选人（默认），带调试记录
bash /root/x2-actor-run/start.sh nozone       # 不用区域（= PR #5 灵巧手的选人），其余一样
bash /root/x2-actor-run/start.sh nozone norec # 不用区域、不记录
bash /root/x2-actor-run/stop.sh               # 停
```

start.sh 起之前会检查这些，任何一条不过就停下、不起：

- 区域文件、参数文件能过正式校验。校验输出记进本趟日志的 `zone_check.txt`。
- 树 `/root/x2-dexhand-wt` 在 `actor-select` 分支。
- 没有别的 `ros2 launch` 在跑。
- 厂商状态是 `STAND_DEFAULT`。

停在「厂商不在 STAND_DEFAULT」：说明上次没有正常退出。先别硬起，查原因。

日志：每趟一个目录 `/root/x2-actor-run/logs/run-<月日-时分秒>-<模式>[-rec]/`。最新一趟的路径写在 `/root/x2-actor-run/LATEST`。

```
tail -f $(cat /root/x2-actor-run/LATEST)/launch.log                 # 启动日志
tail -f $(cat /root/x2-actor-run/LATEST)/rec/actor_state.jsonl      # 选人状态
python3 /root/x2-actor-run/run_read.py $(cat /root/x2-actor-run/LATEST)   # 一趟的选人读数汇总
```

`run_read.py` 打印这几样：

- 锁定/放手的时间线。
- 放手原因：`release:outside` = 出区，`release:occluded` = 被挡，`release:missed` = 检测丢了。
- 人数分布。
- session 状态变化。

记录说明：

- 带记录时，手部节点会把手的小裁图写到 dexhand 的 dump 目录。**可能拍到脸，只留在板上，看完删。**
- 区域标定的截图也有脸，同样不进 git。

---

## 2. 现场标定（笔记本上）

### 2.1 区域文件里有什么

两样东西：

- **地面四边形**（绿色）：四个角 = 人脚的位置。脚在里面才会被选；走出去就放手。
- **左右蓝线**（`crop_x`）：人框**整个**在蓝线外面的人完全不管。蓝线放在旁边围观的人站的地方，比四边形宽一些。

近 = 靠机器人 = 图上偏下。近边两个角离画面底至少 30 px：脚出了画面就看不到了。

### 2.2 一条命令走完

在本目录里执行：

```
bash field_calib.sh
```

1. **采样**（约 75 s）。
   - 按回车开始。开头 5 张清场，画面里不要有人。
   - 屏幕黄字提示「站到【近左角】」时，一个人走过去，脚踩在想要的区域角上，站稳。
   - 之后依次是近右、远右、远左、中间，每个位置 4 张，3 s 一张。
   - 机器人不用起 start.sh，只要头部相机在出图。
2. **叠图**。截图拉回 `~/x2_actor_calib/snap-<时刻>/`，叠成 `composite.jpg`。每帧人的脚标一个红点加帧号。
3. **点角**。弹出窗口，点 6 下：
   1. 左蓝线（只取 x）
   2. 右蓝线（只取 x）
   3. 近左
   4. 近右
   5. 远右
   6. 远左

   按键：`u` 撤销上一点，`r` 重来，`s` 或回车保存（保存时自动跑正式校验，不过会提示），`q` 放弃。
4. **核对**。自动打开 `zone_overlay.png`，终端里印出区域文件。
5. **上传**：输入 `y` 才上传。
   - 先在本机校验，再在 duck 上用装好的那份连同参数文件校验，通过才替换。
   - 旧文件备份成 `actor_zone.yaml.bak-<时刻>`。
   - 两边比对 sha256。
6. **生效**：按提示在 duck 上 `stop.sh` 再 `start.sh`。

### 2.3 分步做（某一步不顺时）

```
bash field_calib.sh snap                          # 只采样拉回，最后一行是本机目录
bash field_calib.sh pick ~/x2_actor_calib/snap-XXXX        # 叠图 + 点角 + 核对，不上传
bash field_calib.sh pick ~/x2_actor_calib/snap-XXXX <某张.jpg>  # 不用叠图，直接在单张上点
bash field_calib.sh push ~/x2_actor_calib/snap-XXXX/actor_zone.yaml   # 校验并上传
```

- **叠图不准**（会场背景有人走动、灯光在变）：红点会乱。直接拿「人站在角上」的那几张单图点，效果一样。
- **想换节奏**：`EMPTY=5 HOLD=4 EVERY=3 bash field_calib.sh`。依次是空场张数、每个位置几张、隔几秒一张。
- **duck 的 ssh 名字不叫 duck**：`DUCK=root@<地址> bash field_calib.sh`。
- **窗口打不开**（cv2 没有 GUI）：用系统的 python，`PY=/usr/bin/python3`（默认就是它）。

### 2.4 标完怎么验

起 `start.sh` 后，每个角都站一下，再从每条边走出去。然后在 duck 上跑：

```
python3 /root/x2-actor-run/run_read.py $(cat /root/x2-actor-run/LATEST)
```

应该看到：

- 人在区里时没有 GRACE，也没有放手。
- 出区后约 2 s 放手，原因 `release:outside`。

只在区里站着却放手了：多半是近边太靠下（脚出画面），或者区域画得太窄。

---

## 3. 调参数（不用构建）

文件 `/root/x2-actor-run/zone/actor_tuning.yaml`。改完 stop/start 生效，start.sh 会先校验。

1 次刷新约 1 s。

| 键 | 默认 | 意思 |
|---|---|---|
| `acquire_updates` | 2 | 候选连续几次刷新得分最高才锁定 |
| `release_updates` | 3 | 连续几次坏刷新（出区/被挡/丢了）才放手 |
| `release_hold_s` | 3.0 | 放手后多久不选人 |
| `center_weight` / `front_weight` | 0.6 / 0.4 | 选人时「居中」和「靠前」的权重 |
| `occlude_overlap` | 0.3 | 被挡判定：身体水平重叠比例；`null` = 不判被挡 |
| `occlude_dy` | 0.05 | 挡人的人脚要低出多少（× 被挡者框高） |
| `foot_half_width` | 0.12 | 脚点半宽（× 框高） |
| `track_keep_updates` | 2 | 别人的轨迹漏检几次后删 |
| `gate_dx` / `gate_dy` / `gate_h_ratio` | null | 关联门限，现在关着 |

---

## 4. 出问题 / 回退

- **选人不对、来不及查**：`stop.sh`，再 `start.sh nozone`。
  - 不用区域，回到 PR #5 的选人方式。
  - 同一棵树，不用构建。
- **新区域不好用，退回上一份**（duck 上）：
  ```
  ls /root/x2-actor-run/zone/            # 找 actor_zone.yaml.bak-<时刻>
  cp /root/x2-actor-run/zone/actor_zone.yaml.bak-<时刻> /root/x2-actor-run/zone/actor_zone.yaml
  ```
  然后 stop/start。
- **实验室那份区域**在 `zone/actor_zone.lab-1024.yaml`。只对实验室的机位有效，会场不能用。
- **start.sh 说校验没过**：看它印出来的 `ERROR` 行（哪个角、差多少），用 `field_calib.sh pick` 重点。
- **双臂卡跳**：多半是 launch 后不到 90 s 就开了镜子。长捂关，等一会儿再开。

---

## 5. 换新板子 / 重新部署

在笔记本上执行：

```
bash field_calib.sh deploy
```

它会把 `duck/` 下的脚本装到 duck 的 `/root/x2-actor-run/`：

- 板上内容不同的旧文件，先备份成 `.bak-<时刻>`。
- `zone/actor_tuning.yaml` 板上没有才装。
- `zone/actor_zone.yaml` 不装，要现场标定。

start.sh 还依赖下面这些，deploy 不管，要先在板上：

- `/root/x2-dexhand-wt`：本仓 `actor-select` 分支，构建过（symlink-install）。
- `/root/IB-ROBOT-X2/venv`：主树的 venv。
- `/root/aimdk/install`：厂商消息。
- `/root/x2-mirror-dds-robot-lan.xml`：DDS 限到机器人网卡（duck 上要有 10.0.1.50）。
- `/root/x2-dexhand-run/`：
  - `models/`、`extern/`（hand_core 运行时加载，**不进 git**）：配置档里写死的路径。
  - `pydeps/`：手部节点的 NPU 运行时。
- `/root/visual-talk-run/rt_watch.py`、`fb_watch.py`：后台只读记录。
- `/root/x2-retarget-run/mc_state.py`：读厂商状态。
