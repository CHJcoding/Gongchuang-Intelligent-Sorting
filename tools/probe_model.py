#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取板端 .bin 的真实输入/输出规格。"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else '/root/vision_ws/yolov5_best.bin'
print('model:', path)

try:
    from hobot_dnn import pyeasy_dnn as dnn
    models = dnn.load(path)
    print('models loaded:', len(models))
    m = models[0]
    print('--- inputs ---')
    for i in m.inputs:
        p = i.properties
        print('  shape=%s h=%s w=%s layout=%s' % (
            p.shape, p.height, p.width, getattr(p, 'layout', '?')))
    print('--- outputs (%d) ---' % len(m.outputs))
    for o in m.outputs:
        print('  shape=%s' % (o.properties.shape,))
except Exception as e:
    print('pyeasy_dnn failed:', type(e).__name__, e)

# 再试 bpu_infer_lib
try:
    import bpu_infer_lib_x5 as bil
    print('--- bpu_infer_lib_x5 ---')
    inf = bil.infer_lib.InferLib(path)
    for name in ('input', 'inputs'):
        pass
    print('InferLib OK; input info:',
          getattr(inf, 'input', None), getattr(inf, 'inputs', None))
except Exception as e:
    print('bpu_infer_lib_x5:', type(e).__name__, e)