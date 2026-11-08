# -*- coding: utf-8 -*-
import glob, os, shutil
p = "/root/vision_ws/cam_watchdog.sh"
baks = sorted(glob.glob(p + ".bak.*"))
assert baks, "没有备份"
shutil.copy2(baks[-1], p)
s = open(p, encoding="utf-8").read()
assert "apply_cam_params" not in s, "恢复后仍有补丁，检查备份"
old = '    log "相机就绪，启动检测节点..."'
new = '\n'.join([
    '    log "应用网页调参保存的摄像头参数（cam_params.json）"',
    '    python3 "$(dirname "$0")/tools/apply_cam_params.py" 2>&1 | sed "s/^/  /"',
    old,
])
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("已用真换行重新打补丁，备份:", baks[-1])
