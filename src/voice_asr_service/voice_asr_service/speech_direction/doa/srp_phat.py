"""保相位 STFT-SRP-PHAT + 段级能量加权累积。

合并了阵列几何与 SRP-PHAT 算法逻辑。算法逻辑与回归基线逐字对齐(回归基线:5 文件 8 段 max err 15°)。

坐标系(阵列坐标系,算法自身约定):
    0°   = +x(右)
    90°  = +y(前)
    180° = -x(左)
    270° = -y(后)
    逆时针为正
方向向量 dir(θ) = (cosθ, sinθ)。

设计要点:
    1. 单帧瞬时互谱 C_ij(f) = X_i(f) * conj(X_j(f)),PHAT 归一化 P_ij = C_ij/(|C_ij|+eps)。
    2. 导向相位矩阵预计算:steering[θ,pair,f] = exp(-j·2π·f·τ_ij(θ)/sr),
       τ_ij(θ) = (pos_j - pos_i)·dir(θ)/c。
    3. per-pair 限频 valid_mask:边相邻对 [300,3400]Hz 全有效,对角对 300~2600Hz(防栅瓣)。
    4. 实时投票 einsum:scores = einsum("apf,pf->a", steering.conj(), P*valid_mask).real。
    5. 段级能量加权累积:acc = Σ(w[k]*scores[k])/Σw,w[k]=rms[k]/max(rms),argmax 出 DOA。
"""

from __future__ import annotations

import numpy as np

# ============================ 阵列几何(合并自阵列几何与声学参数)============================

# 声速 m/s(默认值,ReSpeaker 圆形阵列常温标准值;实例可通过 sound_speed 参数覆盖)
SOUND_SPEED = 343.0

# 4 麦坐标(米),顺序对应 ch1~4。
# 默认值(ReSpeaker USB 4-Mic Array 正方形布局,边长 45.7mm);
# 实例应通过 mic_positions 参数从配置传入,确保配置驱动。
MIC_POSITIONS = np.array(
    [
        [+0.02285, +0.02285],  # Mic1/ch1 右上
        [-0.02285, +0.02285],  # Mic2/ch2 左上
        [-0.02285, -0.02285],  # Mic3/ch3 左下
        [+0.02285, -0.02285],  # Mic4/ch4 右下
    ],
    dtype=np.float64,
)

# 6 对麦(索引 0~3 = ch1~4),顺序与算法基线一致
MIC_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# 对角对的 pair 索引(在 MIC_PAIRS 里的位置):
#   MIC_PAIRS[1]=(0,2) ch1-ch3 对角;MIC_PAIRS[4]=(1,3) ch2-ch4 对角
# 对角对间距大(0.0646m),空间混叠栅瓣频率低(~2654Hz),需单独限频防栅瓣
DIAG_PAIR_IDX = {1, 4}


def dir_vec(theta_deg: float) -> np.ndarray:
    """声源方向单位向量 dir(θ) = (cosθ, sinθ)。

    0°=+x, 90°=+y, 180°=-x, 270°=-y。
    返回 shape (2,) float64。
    """
    rad = np.radians(theta_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)


# ============================ SRP 频段(默认值,实例可覆盖)============================

# 边相邻对 300~3400,对角对 300~2600 防栅瓣
FREQ_LO = 300
FREQ_HI = 3400
DIAG_FREQ_HI = 2600
# 数值稳定小量
EPS = 1e-8


class StftSrpPhat:
    """可变 frame/hop 的保相位 STFT-SRP-PHAT。

    坐标系:0°=+x, 90°=+y。方向向量 (cosθ, sinθ)。
    预计算导向相位矩阵,实时 einsum 投票。

    麦克风坐标、声速、频段门限均从配置传入(配置驱动),不再硬编码。
    """

    def __init__(
        self,
        frame_size: int,
        hop_size: int,
        angles_deg: np.ndarray,
        sample_rate: int = 16000,
        *,
        mic_positions: np.ndarray | None = None,
        sound_speed: float = SOUND_SPEED,
        freq_lo: float = FREQ_LO,
        freq_hi: float = FREQ_HI,
        diag_freq_hi: float = DIAG_FREQ_HI,
    ):
        """
        Args:
            frame_size: STFT 帧长(如 4096)
            hop_size: STFT 步长(如 2048,50% overlap)
            angles_deg: 扫描角度数组(如 np.arange(0, 360, 5))
            sample_rate: 采样率(16000)
            mic_positions: 4 麦坐标 (4, 2) 米,顺序对应 ch1~4;None 用默认 MIC_POSITIONS
            sound_speed: 声速 m/s(默认 343)
            freq_lo: 边相邻对频段下限 Hz(默认 300)
            freq_hi: 边相邻对频段上限 Hz(默认 3400)
            diag_freq_hi: 对角对频段上限 Hz(默认 2600,防空间混叠栅瓣)
        """
        self.N = frame_size
        self.H = hop_size
        self.sr = sample_rate
        self.angles = angles_deg
        self.n_angles = len(angles_deg)
        # Hann 窗(与算法基线一致:np.hanning(N+1)[:-1])
        self.hann = np.hanning(frame_size + 1)[:-1].astype(np.float32)
        self.n_freq = frame_size // 2 + 1
        self.freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

        # 阵列几何与声学参数(从配置传入,默认值仅作 fallback)
        self.mic_positions = np.asarray(mic_positions, dtype=np.float64) if mic_positions is not None else MIC_POSITIONS
        self.sound_speed = float(sound_speed)
        self.freq_lo = float(freq_lo)
        self.freq_hi = float(freq_hi)
        self.diag_freq_hi = float(diag_freq_hi)

        # 预计算导向相位矩阵 + per-pair valid_mask
        # steering: (n_angles, n_pairs, n_freq) 复数
        # valid_mask: (n_pairs, n_freq) 布尔
        self.steering, self.valid_mask = self._precompute()

    def _precompute(self):
        """预计算导向相位矩阵与 per-pair 限频 mask。

        对每个角度 θ、每对麦 (i,j)、每个频点 f:
          理论相位差 = exp(-j·2π·f·τ_ij(θ)/sr)
          其中 τ_ij(θ) = (pos_j - pos_i) · dir(θ) / c(声从 θ 方向来时 j 相对 i 的时延)
          dir(θ) = (cosθ, sinθ)
        SRP 投票用 real(P_ij · conj(steering)),等价于 cos(相位差) 投票。
        """
        n_a = self.n_angles
        n_p = len(MIC_PAIRS)
        n_f = self.n_freq
        steering = np.zeros((n_a, n_p, n_f), dtype=np.complex64)
        valid_mask = np.zeros((n_p, n_f), dtype=bool)

        for pi, (i, j) in enumerate(MIC_PAIRS):
            d = self.mic_positions[j] - self.mic_positions[i]  # (2,)
            # 各角度下 j 相对 i 的时延(秒)
            for ai, theta in enumerate(self.angles):
                rad = np.radians(theta)
                direction = np.array([np.cos(rad), np.sin(rad)])
                tau = float(d.dot(direction)) / self.sound_speed  # 秒
                # 相位差 exp(-jωτ) = exp(-j·2π·f·τ)
                phase = -2j * np.pi * self.freqs * tau
                steering[ai, pi, :] = np.exp(phase).astype(np.complex64)
            # per-pair 限频:对角对上限 diag_freq_hi,边相邻对上限 freq_hi
            is_diag = pi in DIAG_PAIR_IDX
            hi = self.diag_freq_hi if is_diag else self.freq_hi
            valid_mask[pi, :] = (self.freqs >= self.freq_lo) & (self.freqs <= hi)

        return steering, valid_mask

    def stft_4ch(self, audio4: np.ndarray) -> np.ndarray:
        """4ch 时域 → STFT,返回 (n_freq, n_frames, 4) 复数。

        audio4: (T, 4) float32,增强后 4 通道
        Hann 窗 50% overlap(hop = N/2)。
        """
        T = audio4.shape[0]
        U = 1 + (T - self.N) // self.H
        if U < 1:
            U = 1
        spec = np.zeros((self.n_freq, U, 4), dtype=np.complex64)
        for c in range(4):
            for u in range(U):
                s = u * self.H
                frame = audio4[s : s + self.N, c] * self.hann
                spec[:, u, c] = np.fft.rfft(frame, n=self.N)
        return spec

    def compute_all_scores(self, spec4: np.ndarray, w_f: np.ndarray | None = None) -> np.ndarray:
        """逐帧算完整空间谱 scores(不只 argmax),返回 (n_frames, n_angles)。

        用于段级能量加权累积等后处理。
        spec4: (n_freq, n_frames, 4) 复数谱
        w_f  : (n_freq,) 频点可靠性权重,None 表示不加权(全 1)
        """
        F, U, _ = spec4.shape
        if w_f is None:
            w_f = np.ones(F, dtype=np.float32)
        else:
            w_f = np.asarray(w_f, dtype=np.float32)

        all_scores = np.zeros((U, self.n_angles), dtype=np.float32)
        for u in range(U):
            # 6 对的 P_ij(单帧瞬时互谱 + PHAT)
            P = np.zeros((len(MIC_PAIRS), F), dtype=np.complex64)
            for pi, (i, j) in enumerate(MIC_PAIRS):
                cij = spec4[:, u, i] * np.conj(spec4[:, u, j])
                P[pi, :] = cij / (np.abs(cij) + EPS)
            wf_eff = w_f[None, :] * self.valid_mask  # (n_pairs, n_freq)
            # 投票:score[a] = sum_{p,f} real(P[p,f] * conj(steering[a,p,f]))
            # conj(steering) 因为 P_ij = X_i·conj(X_j),理论相位差方向相反
            all_scores[u] = np.einsum("apf,pf->a", self.steering.conj(), P * wf_eff).real
        return all_scores

    def locate_seg_energy_weighted(
        self,
        spec4: np.ndarray,
        frame_sel: np.ndarray,
        rms_per_frame: np.ndarray,
        w_f: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        """段级能量加权累积谱 DOA(推荐主方法)。

        把段内多帧 scores 按每帧 RMS 加权累积,再 argmax。
        声音大的帧(高信噪比、互谱相位准)主导,噪声帧(RMS 小、DOA 乱飘)降权。

        spec4: (n_freq, n_frames, 4) 复数谱
        frame_sel: 段内帧索引数组
        rms_per_frame: 每帧 RMS(全段,长度=n_frames),用于取权重
        w_f: 频点权重,None=不加权
        return: (段DOA角度, 累积谱)
        """
        all_scores = self.compute_all_scores(spec4, w_f)  # (n_frames, n_angles)
        seg_scores = all_scores[frame_sel]  # (n_seg_frames, n_angles)
        rms_seg = np.asarray(rms_per_frame[frame_sel], dtype=np.float32)
        rms_seg = np.maximum(rms_seg, 1e-8)
        w = rms_seg / (rms_seg.max() + 1e-8)  # 归一化到 [0,1]
        acc = (seg_scores * w[:, None]).sum(0) / (w.sum() + 1e-8)  # (n_angles,)
        return int(self.angles[int(np.argmax(acc))]), acc


__all__ = [
    "StftSrpPhat",
    "MIC_POSITIONS",
    "MIC_PAIRS",
    "DIAG_PAIR_IDX",
    "SOUND_SPEED",
    "dir_vec",
    "FREQ_LO",
    "FREQ_HI",
    "DIAG_FREQ_HI",
    "EPS",
]
