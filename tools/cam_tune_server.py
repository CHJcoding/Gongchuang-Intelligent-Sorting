#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摄像头参数网页调参工具（对齐官方标定教程 §2.2「调节摄像头参数使图像清晰」）。

板子无显示器，跑不了教程里的 OpenCV 弹窗，这里用 stdlib HTTP 提供同款能力：
    source /opt/tros/humble/setup.bash
    python3 cam_tune_server.py            # 默认 http://<板IP>:8002/
页面滑杆实时写 v4l2 控件，同屏显示清晰度 / HSV 中位 / 与基线的漂移，便于判断
"调完是否真的更清楚"以及"判色阈值要跟着改多少"。
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

BOUNDARY = "gcjpegframe"
PARAM_FILE = "/root/vision_ws/cam_params.json"
DEV = "/dev/video0"
# 允许网页改的控件（其余一律不理会，防止误改曝光模式导致画面发黑）
ALLOWED = ["brightness", "contrast", "saturation", "hue", "gamma", "gain",
           "sharpness", "backlight_compensation", "white_balance_automatic",
           "auto_exposure", "exposure_time_absolute"]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>摄像头调参</title>
<style>
body{margin:0;background:#141414;color:#ddd;font:13px/1.5 monospace}
#wrap{display:flex;flex-wrap:wrap;gap:14px;padding:10px}
#left{flex:0 0 auto} img{width:640px;max-width:96vw;display:block;border:1px solid #333}
#right{flex:1 1 320px;min-width:300px;max-width:520px}
.row{margin:7px 0} .nm{display:inline-block;width:180px;color:#9cf}
input[type=range]{width:150px;vertical-align:middle}
.val{display:inline-block;width:46px;text-align:right;color:#ff9}
#m{background:#1d1d1d;border:1px solid #333;padding:8px;white-space:pre;font-size:12px}
button{background:#2a4;border:0;color:#fff;padding:6px 10px;margin:4px 6px 0 0;cursor:pointer;font:13px monospace}
button.g{background:#357} button.r{background:#a33}
h3{margin:14px 0 4px;color:#8cf;font-size:14px}
</style></head><body>
<div id="wrap">
 <div id="left"><img id="im" src="/stream" alt="stream">
  <div id="m">加载中…</div>
 </div>
 <div id="right">
  <h3>图像参数（实时生效）</h3>
  <div id="ctrls"></div>
  <button onclick="mark()">记基线</button>
  <button class="g" onclick="save()">写入开机参数</button>
  <button class="r" onclick="reset()">恢复驱动默认</button>
  <div id="tip" style="margin-top:10px;color:#888;font-size:12px">
   注意：对比度/饱和度/伽马一动，HSV 整体漂移，颜色阈值(S≥40 等)要跟着重标。
  </div>
 </div>
</div>
<script>
var CUR=null, BASE=null;
function fmt(v,d){return (v===undefined||v===null||v==='')?'-':v;}
function build(st){
  var box=document.getElementById('ctrls'); box.innerHTML='';
  st.controls.forEach(function(c){
    var d=document.createElement('div'); d.className='row';
    d.innerHTML='<span class="nm">'+c.name+'</span>'+
      '<input type="range" min="'+c.min+'" max="'+c.max+'" step="'+c.step+'" value="'+c.value+
      '" oninput="setp(this,\''+c.name+'\')">'+
      '<span class="val" id="v_'+c.name+'">'+c.value+'</span>';
    box.appendChild(d);
  });
}
function setp(el,name){
  document.getElementById('v_'+name).textContent=el.value;
  fetch('/set?name='+encodeURIComponent(name)+'&value='+encodeURIComponent(el.value))
    .then(function(r){return r.json();}).then(function(j){ if(j.error){alert(j.error);} poll(); });
}
function mark(){ fetch('/baseline').then(function(r){return r.json();}).then(function(j){BASE=j.metrics; poll(); }); }
function save(){ fetch('/save').then(function(r){return r.json();}).then(function(j){
  document.getElementById('tip').innerHTML='已写入 '+j.path+'<br>注意：对比度/饱和度/伽马一动，HSV 整体漂移，颜色阈值要跟着重标。'; }); }
function reset(){ if(!confirm('恢复驱动默认参数？')) return;
  fetch('/reset').then(function(r){return r.json();}).then(function(){ poll(true); }); }
function poll(rebuild){
  fetch('/state').then(function(r){return r.json();}).then(function(st){
    if(rebuild || !CUR) build(st);
    CUR=st;
    var m=st.metrics||{};
    var b=BASE||{};
    function d(k){ if(m[k]===undefined) return '-'; if(b[k]===undefined) return ''; return ' ('+(m[k]-b[k]>=0?'+':'')+(m[k]-b[k]).toFixed(1)+')'; }
    document.getElementById('m').textContent =
      '清晰度 lap   : '+fmt(m.lap)+d('lap')+'   ← 越大越锐\n'+
      '亮度 V 中位  : '+fmt(m.v)+d('v')+'\n'+
      '饱和度S中位  : '+fmt(m.s)+d('s')+'\n'+
      '色相 H 中位  : '+fmt(m.h)+d('h')+'\n'+
      '── 高饱和(S>=40)物料区 ──\n'+
      'H / S / V    : '+fmt(m.c_h)+' / '+fmt(m.c_s)+' / '+fmt(m.c_v)+'\n'+
      '帧率/分辨率  : '+fmt(m.fps)+' fps  '+fmt(m.size)+'\n'+
      '曝光         : '+fmt(st.expo)+'  增益 '+fmt(st.gain)+'\n'+
      '基线         : '+(BASE?('已记（lap '+fmt(BASE.lap)+' / H '+fmt(BASE.h)+' / S '+fmt(BASE.s)+' / V '+fmt(BASE.v)+'）'):'未记');
  }).catch(function(){});
}
poll(true); setInterval(function(){poll(false);},1500);
</script></body></html>"""


def sh(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=5)


def read_ctrls():
    """v4l2-ctl --list-ctrls 解析成 [{name,min,max,step,default,value}]。"""
    out = sh(["v4l2-ctl", "-d", DEV, "--list-ctrls"]).stdout
    items = []
    for line in out.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\([a-z]+\)\s*:\s*(.*)", line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        def g(k):
            mm = re.search(k + r"=(-?\d+)", rest)
            return int(mm.group(1)) if mm else None
        v = g("value")
        if v is None:
            continue
        items.append({"name": name, "min": g("min"), "max": g("max"),
                      "step": g("step") or 1, "default": g("default"), "value": v})
    return [c for c in items if c["name"] in ALLOWED]


def set_ctrl(name, value):
    if name not in ALLOWED:
        return False, "不允许改这个控件"
    return sh(["v4l2-ctl", "-d", DEV, "-c", "%s=%s" % (name, value)]).returncode == 0, ""


class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.metrics = {}
        self.stamp = 0.0

    def set(self, jpeg, metrics):
        with self.lock:
            self.jpeg = jpeg
            self.metrics = metrics
            self.stamp = time.time()

    def get(self):
        with self.lock:
            return self.jpeg

    def snap(self):
        with self.lock:
            return dict(self.metrics), (time.time() - self.stamp if self.stamp else -1)


def measure(img, dt):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 只统计"非背景"区域：亮度前 60% 的像素，避免大片台面把中位数拉平
    v = hsv[:, :, 2]
    thr = float(np.percentile(v, 40))
    mask = v >= thr
    out = {}
    for i, k in enumerate("hsv"):
        ch = hsv[:, :, i][mask]
        out[k] = float(np.median(ch)) if ch.size else 0.0
    # 高饱和像素（S>=40）才是"物料的颜色"，阈值标定时看这一组
    smask = hsv[:, :, 1] >= 40
    for i, k in enumerate("hsv"):
        ch = hsv[:, :, i][smask]
        out["c_" + k] = float(np.median(ch)) if ch.size else -1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out["lap"] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    out["size"] = "%dx%d" % (img.shape[1], img.shape[0])
    out["fps"] = round(1.0 / dt, 1) if dt > 0 else 0.0
    for k in ("h", "s", "v", "lap", "c_h", "c_s", "c_v"):
        out[k] = round(out[k], 1)
    return out


class Tune(Node):
    def __init__(self, latest):
        super().__init__("gc_cam_tune")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CompressedImage, "/image", self.cb, qos)
        self.latest = latest
        self.t_last = 0.0

    def cb(self, msg):
        now = time.time()
        dt = now - self.t_last if self.t_last else 0.0
        self.t_last = now
        buf = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return
        self.latest.set(bytes(msg.data), measure(img, dt))


def snapshot_initial():
    """驱动自报的 default 不可靠（实测 brightness default=0 但当前在跑 50），
    所以「恢复默认」以服务启动时的真实值为准。"""
    return {c["name"]: c["value"] for c in read_ctrls() if c.get("min") is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()

    latest = Latest()
    baseline = {"m": None}
    initial = snapshot_initial()
    rclpy.init()
    node = Tune(latest)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/state":
                metrics, age = latest.snap()
                ct = {c["name"]: c["value"] for c in read_ctrls()}
                return self._send(200, json.dumps({
                    "controls": read_ctrls(), "metrics": metrics, "age_s": round(age, 2),
                    "expo": ct.get("exposure_time_absolute"), "gain": ct.get("gain"),
                    "auto_exposure": ct.get("auto_exposure"),
                }))
            if u.path == "/set":
                name = (qs.get("name") or [""])[0]
                val = (qs.get("value") or [""])[0]
                ok, err = set_ctrl(name, val)
                return self._send(200, json.dumps({"ok": ok, "error": err,
                                                   "controls": {c["name"]: c["value"] for c in read_ctrls()}}))
            if u.path == "/baseline":
                m, _ = latest.snap()
                baseline["m"] = m
                return self._send(200, json.dumps({"metrics": m}))
            if u.path == "/save":
                cur = {c["name"]: c["value"] for c in read_ctrls()}
                with open(PARAM_FILE, "w", encoding="utf-8") as f:
                    json.dump(cur, f, ensure_ascii=False, indent=2)
                return self._send(200, json.dumps({"ok": True, "path": PARAM_FILE, "values": cur}))
            if u.path == "/reset":
                bad = []
                for name, val in initial.items():
                    ok, _ = set_ctrl(name, val)
                    if not ok and name != "exposure_time_absolute":
                        bad.append(name)
                return self._send(200, json.dumps({"ok": True, "failed": bad,
                                                   "controls": {c["name"]: c["value"] for c in read_ctrls()}}))
            if u.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        jpeg = latest.get()
                        if jpeg:
                            self.wfile.write(("--" + BOUNDARY + "\r\nContent-Type: image/jpeg\r\n"
                                              "Content-Length: " + str(len(jpeg)) + "\r\n\r\n").encode())
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    return
            return self._send(404, json.dumps({"error": "no route"}))

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    print("[gc_cam_tune] HTTP 调参页已就绪 http://0.0.0.0:%d/" % args.port, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
