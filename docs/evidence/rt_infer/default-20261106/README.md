# 默认 RT-Infer 集成板端证据

七个 case 使用同一 boot、固定 release 和 90 秒有效窗口。完整方法见
[验证报告](../../../rt_infer_default_integration_validation.md)。

- `cases/` 保留逐次 CSV、就绪身份、数值检查、五进程退出及停止状态。
- `final-v3-independent-audit.json` 由监督端按整数时间戳独立重算。
- `audit_final_v3.py` 记录监督端复算过程，脚本中的本机路径需按下载位置调整。
- `native-recovery-final.json` 记录最终重启后的官方运行库及四模型数值恢复。
- `source-v3-complete-sha256.txt` 固定实际测量源码，后续只修改文档和此证据目录。
- `test-baseline-comparison.json`、`navigation-hang-comparison.json` 保留未通过的旧测试对照。

CSV 入库仅把 CRLF 换为 LF，字段与时间戳不变；未经处理的原始归档 SHA256
记录在验证报告中。证据中的进程、目录和单调时钟值是当次事实，不代表当前运行状态。

软调度改善 PEAR 尾延迟，同时明显降低 TTS/VL 后台进展，不构成硬实时或公平性保证。
