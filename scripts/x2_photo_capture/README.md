# X2 照镜子拍照 —— 上位机接收端和 .5 上的地址工具

照镜子每一轮（进 MIRRORING 到出 MIRRORING）挑一张最好的照片，连同人框发给上位机（现场是跑 AIGC 的 Windows 电脑，开发时是 Ubuntu 笔记本）。拍照是一个单独的节点 `mirror_photo`，镜像不等它：上位机不在线、网断了、盘满了，镜像照常。

本目录三样东西：

- `receiver.py`：上位机上跑的接收端。只用 Python ≥ 3.8 标准库，Windows 和 Ubuntu 同一份。
- `photo_target.sh`：.5 上改上位机地址。
- `send_test.py`：.5 上不起 ROS、发合成测试图，查通不通。

---

## 0. 现场速查（按顺序做）

1. 上位机起接收端（第 1 节）。屏幕会打印本机地址，以及 .5 上要执行的那一行。
2. Windows 防火墙放行 TCP 8765（第 1 节）。
3. .5 上写入上位机地址（第 2 节）：
   ```
   bash <.5 上的树>/scripts/x2_photo_capture/photo_target.sh 192.168.3.24
   ```
   看到「/health 正常」才算通。
4. 发 3 张测试图（第 3 节），上位机窗口里应各打出一行。
5. 照常起整套、开镜子。每收一轮，上位机窗口打一行 `photo <轮次id> ... -> <轮次id>.jpg`。

---

## 1. 上位机：起接收端

开发阶段**一律加 `--discard`**：只记账本，不存照片字节（照片里有人像）。

```
python3 receiver.py --discard                          # Ubuntu，开发
python receiver.py --inbox D:\x2_photos                # Windows，现场：存照片
python receiver.py --inbox D:\x2_photos --crop         # 另存按人框裁的 <轮次id>_crop.jpg（要 pip install pillow）
python receiver.py --inbox D:\x2_photos --on-photo "python aigc.py {path}"
python receiver.py --inbox D:\x2_photos --allow 192.168.3.10   # 只收 .5 发来的，其它来源 403
```

| 参数 | 含义 |
|---|---|
| `--inbox` | 照片和账本放哪，默认当前目录下 `x2_photos` |
| `--discard` | 只记账本 `ledger.jsonl`，不存照片；`--crop`、`--on-photo` 随之不起作用 |
| `--crop` | 人框四边各外扩框宽高的 15 %，裁出来另存。没装 Pillow 只提示一次，原图照存 |
| `--on-photo` | 每张新照片落盘后，在后台执行一条命令（不等它跑完，不影响回执）。可用 `{path}` 原图、`{meta}` 元数据、`{crop}` 裁图（没有则为空）、`{round_id}`，会自动加引号。退出码非 0 只记一行 |
| `--allow` | 只收这个来源地址，可写多次。不写就谁都收 |
| `--port` | 默认 8765 |
| `--max-mb` | 单张上限，默认 5 MB |

防火墙：

- Windows（管理员 PowerShell，一次即可）：
  ```
  netsh advfirewall firewall add rule name="x2-photo" dir=in action=allow protocol=TCP localport=8765
  ```
- Ubuntu：只有 `sudo ufw status` 显示 active 时才需要 `sudo ufw allow 8765/tcp`。

inbox 里的文件：

```
<轮次id>.json          元数据：bbox [x1, y1, x2, y2]（原图像素）、width/height、quality、source/label、score、收到时间、来源地址 ...
<轮次id>.jpg           相机给的原始 JPEG（半分辨率），一字节没改
<轮次id>_crop.jpg      仅 --crop
ledger.jsonl           账本：每收到一次一行（包括重发），--discard 也记
```

先写 `.json` 再写 `.jpg`，两个都是写临时文件再改名。所以盯着 `*.jpg` 的工作流看到图时，元数据一定已经在了。

`quality` 字段：`normal` 正常挑出的；`fallback` 本轮没有候选、或挑选出错，用的是兜底帧；`recovered` .5 上的节点崩过，重启后补发的上一轮；`test` 是 `send_test.py` 发的合成图，AIGC 应跳过。

---

## 2. .5：写上位机地址

```
bash photo_target.sh                    # 看当前地址、outbox 排队张数，并访问一次 /health
bash photo_target.sh 192.168.3.24       # 写入（默认端口 8765）
bash photo_target.sh 192.168.3.24:9000  # 带端口
```

- 地址写进 `/root/x2-feat-photo-capture-run/photo_target`。节点每次发图前都会重新读一次，**改完不用重启**，排队中的照片会发往新地址。
- 填的是上位机在同一张网上的地址，不是 .5 自己的。上位机起 `receiver.py` 时打印的就是它。
- `/health` 不通时照样写入，并给黄字提示。照片留在 outbox 里，上位机一起来就补发。地址格式不对则不写。
- 这个脚本只读路由表（UDP connect 不发包），不改 .5 的任何网络设置。
- 运行目录不在默认位置时，用 `PHOTO_RUN=<目录> bash photo_target.sh ...`。

---

## 3. .5：发测试图

```
python3 send_test.py health                 # 用目标文件里的地址
python3 send_test.py send -n 3              # 发 3 张 64x48 合成色块（没有人像）
python3 send_test.py send 192.168.3.24 -n 1 # 不看目标文件，直接给地址
```

发送用的就是节点里的那套代码（`mirror_photo.uploader.post_photo`），从本仓 `src` 导入，不需要 source ROS。只有上位机回了 200、而且回执里的 sha256 对得上，才算「已确认」。

退出码：0 全部确认；1 对方没回或没确认；2 地址写错。

---

## 4. 照片丢不了：靠什么

- **.5 上的 outbox**（`/root/x2-feat-photo-capture-run/photos/outbox/`）：每轮挑好的照片先落盘再发。只有收到上位机的 sha256 回执才删，否则一直重发（1、2、5、10、30 s 退避），新的先发。最多留 200 张、100 MB；超了删最旧的。盘剩不到 2 GB 时不落盘，只放在内存里发。
- **一轮进行中的暂存 `current.jpg`**：节点或板子崩了，重启后把它当作上一轮补发（`quality: recovered`）。
- **上位机按账本去重**：同一轮次、同样字节的重发只回执、不多存。同一轮次 id 来了不同字节，另存为 `<轮次id>_<sha 前 8 位>`。重启接收端后照样认得。
- 上位机存盘失败（盘满等）时回 500，不算确认，.5 会继续重发。

对账：

- .5 上 `bash photo_target.sh` 显示「outbox 排队 0 张」，就是全部都已收到回执。
- 节点状态（.5 上，source 过环境之后）：
  ```
  ros2 topic echo --once /mirror_photo/status
  ```
  `upload.delivered` 是本次启动以来确认的张数，`upload.last_error` 是最近一次失败的原因，`outbox` 是排队情况。
- 上位机 `ledger.jsonl` 里 `"duplicate": false` 的行数，就是收到的照片数。

---

## 5. 关掉拍照

开关是 `mirror_session` 块里的 `photo.enabled`：`aimdk_x2_voice_mirror_all.yaml` 里是 `true`，grasp 和 summit 继承它。

在任一入口 yaml 的 `mirror_session` 下加一行，就只关这个入口：

```yaml
robot:
  mirror_session:
    photo: {enabled: false}
```

关掉后不起拍照节点，launch 和没有这个功能之前一样。

---

## 6. 协议（给要自己写接收端的人）

- `POST /x2/photo`：请求体是原始 JPEG，元数据是请求头 `X-X2-Photo-Meta` 里的一个 JSON 对象（纯 ASCII）。必须有 `round_id`（字母数字和 `. _ -`，不以点开头，≤ 80 字符）；有 `sha256` 时必须和请求体一致。
- 存好后回 200，正文为 `{"ok": true, "sha256": "<请求体的 sha256>", ...}`。.5 只认 sha256 对得上的 200，其它一律重发。
- `GET /health` 回 `{"ok": true, "service": "x2-photo-receiver", ...}`，`send_test.py` 和 `photo_target.sh` 用它判断通不通。
- 没有别的路径：不能下载照片，也不能列目录。

拍照节点内部的触发、评分和外部事件接口（`/mirror_photo/trigger`）见 `src/embodied_common/embodied_common/mirror_photo_contract.py`。
