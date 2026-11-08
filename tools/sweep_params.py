# -*- coding: utf-8 -*-
"""扫几组参数，量 lap 与高饱和区 HSV 漂移，最后恢复初始值。"""
import json, time, urllib.request

B = "http://127.0.0.1:8002"

def get(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=5).read().decode())

def setp(n, v):
    return get("/set?name=%s&value=%s" % (n, v))

init = {c["name"]: c["value"] for c in get("/state")["controls"]}
print("初始:", {k: init[k] for k in ("brightness", "contrast", "saturation", "gamma", "gain", "sharpness")})
print("%-28s %8s %7s %7s %7s" % ("参数", "lap", "H", "S", "V"))
def row(tag):
    time.sleep(2.2)
    m = get("/state")["metrics"]
    print("%-28s %8.1f %7.1f %7.1f %7.1f" % (tag, m["lap"], m["c_h"], m["c_s"], m["c_v"]))

row("基线")
for g in (140, 180):
    setp("gamma", g); row("gamma=%d" % g)
setp("gamma", init["gamma"])
for c in (48, 16):
    setp("contrast", c); row("contrast=%d" % c)
setp("contrast", init["contrast"])
for s in (90, 40):
    setp("saturation", s); row("saturation=%d" % s)
setp("saturation", init["saturation"])
setp("sharpness", 0); row("sharpness=0")
setp("sharpness", 6); row("sharpness=6")
setp("sharpness", init["sharpness"])
setp("auto_exposure", 1); setp("exposure_time_absolute", 250); row("手动曝光 250")
setp("exposure_time_absolute", 900); row("手动曝光 900")
# 恢复
for k, v in init.items():
    try: setp(k, v)
    except Exception as e: pass
time.sleep(1)
print("已恢复初始参数")
