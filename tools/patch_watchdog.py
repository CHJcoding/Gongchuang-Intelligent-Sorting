# -*- coding: utf-8 -*-
import shutil, time
p = "/root/vision_ws/cam_watchdog.sh"
s = open(p, encoding="utf-8").read()
if "apply_cam_params" in s:
    print("已经打过补丁")
else:
    old = '    log "相机就绪，启动检测节点..."'
    assert old in s, "没找到 启动检测节点 那行"
    new = ('    log "应用网页调参保存的摄像头参数（cam_params.json）"\\n'
           '    python3 "$(dirname "$0")/tools/apply_cam_params.py" 2>&1 | sed "s/^/  /"\\n'
           + old)
    shutil.copy2(p, p + ".bak." + time.strftime("%m%d_%H%M"))
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    print("已插入 apply_cam_params 调用")
