"""EHM 的替代品：官方中性 SMPL-X 骨架上的刚性前向运动学。

为什么要换掉 EHM：`atc` 转不动那张 136 MB 的图，昇腾上没有 EHM。所以必须
拿一个纯 numpy 的东西顶上，而且必须顶得住——不是"差不多"。

它是什么：SMPL-X 的 joints 本来就是 `J_regressor @ v_template` 再走一遍
标准 FK。把 betas 钉死在 0（中性人体），`J` 就是一张常数表，55 个 offset
加一个骨盆静止位置，2.6 KB。剩下的只有 54 次 3x3 矩阵乘。

它和 EHM 差在哪（tools/build_official_skeleton.py 实测，clip02 60 帧，
betas=0，绝对位置 mm）：

    骨盆+脊柱 0,3,6,9     mean 0.000   max 0.001
    锁骨 13,14           mean 0.000   max 0.000
    左手 25-39           mean 0.000   max 0.000      <- 精确
    右手 40-54           mean 0.000   max 0.001      <- 精确
    颈/头 12,15          mean 65.5    max 101.2
    肩 16,17             mean 35.1    max 49.7
    肘 18,19             mean 33.6    max 45.6
    腕 20,21             mean 7.1     max 9.8

零的那些是**恒等**，不是"误差小"：那些关节 EHM 也只是在做同一套 FK。
非零的七个是 EHM 导出图里那几个 ScatterND——它从形变后的网格重新回归了
颈/肩/肘/腕，即"这个人的体型"。中性化本来就是要把体型拿掉，所以这几毫米
是这次改动的**内容**，不是它的**误差**。判据是渲染出来的主观效果，不是 mm。

不做人脸：22 下巴、23/24 眼球在这里恒等——导出的 EHM 图里 jaw_pose /
eye_pose 本来就传的 None。55-144（FLAME 面部 landmark 与附加点）填 **0**，
不是 NaN：`smplx2robot.source.validate()` 对整个 points 数组跑
`np.isfinite(...).all()`，NaN 会让整段 clip 直接不合法。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SKELETON_NPZ = Path(__file__).resolve().parent / "data" / "smplx_neutral_skeleton.npz"

NJ = 55          # SMPL-X 骨架：0 骨盆 | 1-21 身体 | 22 下巴 23/24 眼 | 25-54 双手
NPOINTS = 145    # EHM 输出的点数，下游 PearFrame.joints 是定长 435=145*3


class NeutralFK:
    """一帧 decoded 参数 -> joints[145,3]。非线程安全：内部复用缓冲区。"""

    def __init__(self, table: Path | None = None):
        self.path = Path(table) if table is not None else SKELETON_NPZ
        if not self.path.is_file():
            raise FileNotFoundError(
                f"中性骨架表不在：{self.path}\n"
                f"重新烘焙：python3 tools/build_official_skeleton.py")
        with np.load(self.path, allow_pickle=False) as data:
            self.offsets = data["offsets"].astype(np.float64)      # (55,3) 父坐标系
            self.parents = data["parents"].astype(np.int64)        # (55,)  parents[0]=-1
            self.root = data["root"].astype(np.float64)            # (3,)   静止骨盆
            self.source = str(data["source"]) if "source" in data else "?"
        if self.offsets.shape != (NJ, 3) or self.parents.shape != (NJ,):
            raise ValueError(f"{self.path} 形状不对：{self.offsets.shape} / {self.parents.shape}")

        # 复用的缓冲区。`_local` 只在 0 / 1:22 / 25:40 / 40:55 被写，
        # 22,23,24 从此恒等——下巴和眼球不动，是故意的。
        self._local = np.tile(np.eye(3), (NJ, 1, 1))
        self._rot = np.empty((NJ, 3, 3))
        self._out = np.zeros((NPOINTS, 3))                          # 55: 之后恒为 0

    def __call__(self, dec: dict) -> np.ndarray:
        """dec 是 `decode_outputs()` 的输出（带 batch 轴）。返回 (145,3) float32。"""
        local, rot, out = self._local, self._rot, self._out
        local[0] = dec["global_orient"].reshape(3, 3)
        local[1:22] = dec["body_pose"].reshape(21, 3, 3)
        local[25:40] = dec["left_hand_pose"].reshape(15, 3, 3)
        local[40:55] = dec["right_hand_pose"].reshape(15, 3, 3)

        # 骨盆落在**静止骨盆位置**，不是原点——EHM 也是这样，实测对到 3e-8 m。
        rot[0] = local[0]
        out[0] = self.root
        for j in range(1, NJ):
            p = self.parents[j]          # SMPL-X 的 kintree 保证 p < j
            rot[j] = rot[p] @ local[j]
            out[j] = out[p] + rot[p] @ self.offsets[j]
        return out.astype(np.float32)    # astype 顺手拷贝，缓冲区可以安全复用
