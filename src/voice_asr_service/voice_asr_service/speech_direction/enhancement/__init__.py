"""enhancement 子包:语音增强算法(FullSubNet)。

只放通用增强算法,不包含 ROS、设备、VAD、DOA 和文件写入。

关于线程:本子包的增强器与执行器内部使用 ``threading.RLock`` / ``threading.Lock``
保护 OM/Device state 的 ping-pong 切换与 ``close`` 生命周期,这是"推理路径互斥"
而非"启动线程"——增强器不创建工作线程,worker 线程由上层 ``SpeechDirectionRuntime``
驱动。原 docstring 的"不包含线程"措辞会让人误以为连锁都用不得,故改为只排除
"自建线程",锁作为状态保护保留。
"""

from .factory import build_stateful_fullsubnet
from .fullsubnet import FullSubNetEnhancer
from .fullsubnet_stateful import StatefulFullSubNetEnhancer

__all__ = ["FullSubNetEnhancer", "StatefulFullSubNetEnhancer", "build_stateful_fullsubnet"]
