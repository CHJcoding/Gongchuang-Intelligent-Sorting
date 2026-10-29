#!/bin/bash
# run_det.sh —— 由 cam_watchdog.sh 调用（不要手工跑第二条：相机是独占的）。
#
# 历史：这里原来起的是官方 dnn_node_example launch。
# 换成 v8 模型后它的 C++ parser 读 float32 的 reg 输出会 SIGSEGV，且业务节点
# 自己推理（USE_LOCAL_DETECTION=True）根本不需要它，所以改成只起相机 + 预览。
exec /root/vision_ws/cam_only.sh