# 离线回归基线:6 文件段级 DOA 误差

回归门限 **每段误差 ≤ 15°**(5° 步长量化内)。下表为本地实测基线证据,
对应测试 `test_offline_regression.py::TestOfflineRegression`。

## 基线数据

| 文件 | GT 角度 | 段数 | 状态 |
|---|---|---|---|
| sound_16s_180.flac | [180] | 1 | ✅ |
| sound_10s_90.flac | [90] | 1 | ✅ |
| sound_18s_0_0.flac | [0, 0] | 2 | ✅ |
| sound_10s_90_0.flac | [90, 0] | 2 | ✅ |
| sound_9s_0_90.flac | [0, 90] | 2 | ✅ |
| sound_19s_90_90_90_90.flac | [90,90,90,90] | 4 | ✅ |

上表每条段级 DOA 输出与 GT 的圆周误差均 ≤ 15°,满足回归门限。

## 汇总(6 文件统一)

- 文件数:6,统计段数:12(GT 中非 None 段)
- 通过段数:12,通过率:100%
- 所有段误差 ≤ 15°(门限)
- `sound_9s_0_90.flac` 第 3 段为碎段(无 GT),不纳入统计

## 复现方式

```bash
# 需先下载模型:./scripts/download_speech_direction_models.sh
# 音频夹具随仓库提供(本目录 FLAC 无损压缩,~6MB),无需额外下载
python -m pytest test/speech_direction/test_offline_regression.py -v -s
```

模型缺失时测试整体 skip(不阻塞 CI);音频缺失时同理。
