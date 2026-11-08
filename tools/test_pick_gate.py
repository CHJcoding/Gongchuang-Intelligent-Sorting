# -*- coding: utf-8 -*-
"""工作空间限位 + 稳定性选点的离线自检（不启动 ROS）。"""
import ast
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "vision_sorter", "sort_vision_node.py")
for p in (os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "vision_sorter")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C

FAIL = []
OK = [0]


def check(name, cond, extra=""):
    if cond:
        OK[0] += 1
    else:
        FAIL.append("%s %s" % (name, extra))


src = open(SRC, encoding="utf-8").read()
tree = ast.parse(src)

WANT = ("_target_xy_mm", "_in_workspace")
found = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        found[node.name] = ast.get_source_segment(src, node)

check("源码里有 _target_xy_mm / _in_workspace", set(found) == set(WANT), str(sorted(found)))
check("有工作空间预过滤（越界打回）", "打回重新识别：越界目标" in src)
check("有抓取点硬闸（精修后越界也不发）", "抓取点越界，打回重新识别" in src)
check("有稳定性选点（连续帧确认）", "self._cand_hits < need" in src)
check("工作空间默认 -150~150",
      (C.WORKSPACE_X_MIN, C.WORKSPACE_X_MAX, C.WORKSPACE_Y_MIN, C.WORKSPACE_Y_MAX) == (-150.0, 150.0, -150.0, 150.0),
      str((C.WORKSPACE_X_MIN, C.WORKSPACE_X_MAX, C.WORKSPACE_Y_MIN, C.WORKSPACE_Y_MAX)))
check("稳定性参数 STABLE_PICK_FRAMES=2 / 30px",
      (C.STABLE_PICK_FRAMES, C.STABLE_PICK_PX) == (2, 30.0),
      str((C.STABLE_PICK_FRAMES, C.STABLE_PICK_PX)))


class _Log(object):
    def debug(self, *a, **k):
        pass


class Fake(object):
    """只借两个方法的语义：毫米换算 + 是否在工作空间内。"""

    def __init__(self):
        self.get_logger = lambda: _Log()


class _Cam(object):
    @staticmethod
    def undistort_point(x, y):
        return (x, y)


def _pixel_to_robot(x, y):
    # 1px = 1mm 的假映射，只为验证闸门逻辑
    return (x - 100.0, y - 100.0, 0.0)


g = {"C": C, "camera_model": _Cam, "pixel_to_robot": _pixel_to_robot, "time": __import__("time")}
for name in WANT:
    exec(textwrap.dedent(found[name]), g)
    setattr(Fake, name, g[name])

f = Fake()
# bbox 中心 = (px,py) → mm = (px-100, py-100)
def mm(px, py):
    return f._target_xy_mm({"bbox": [px - 5, py - 5, px + 5, py + 5]})

check("框中心 → 毫米换算", mm(200, 150) == (100.0, 50.0), str(mm(200, 150)))
check("中心点在内", f._in_workspace(0.0, 0.0))
check("边界 150 算在内", f._in_workspace(150.0, -150.0))
check("X 超 0.1 就出界", not f._in_workspace(150.1, 0.0))
check("Y 超下限出界", not f._in_workspace(0.0, -150.5))
check("远点出界", not f._in_workspace(400.0, 0.0))
check("换算后的越界目标会被判出界", not f._in_workspace(*mm(400, 200)))

# 关掉限位（设为 None）时应全部放行
old = (C.WORKSPACE_X_MIN, C.WORKSPACE_X_MAX, C.WORKSPACE_Y_MIN, C.WORKSPACE_Y_MAX)
try:
    C.WORKSPACE_X_MIN = C.WORKSPACE_X_MAX = C.WORKSPACE_Y_MIN = C.WORKSPACE_Y_MAX = None
    check("关闭限位后不拦", f._in_workspace(9999.0, -9999.0))
finally:
    (C.WORKSPACE_X_MIN, C.WORKSPACE_X_MAX, C.WORKSPACE_Y_MIN, C.WORKSPACE_Y_MAX) = old

print("通过 %d 项，失败 %d 项" % (OK[0], len(FAIL)))
for x in FAIL:
    print("  [失败] " + x)
print("全部通过" if not FAIL else "有失败项")
sys.exit(0 if not FAIL else 1)
