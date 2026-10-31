#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓帧工具（板端网页版）：把现场真实帧按类别存成数据集。

为什么不用 PC 摄像头或以前录的视频：
  训练集必须和部署时的相机/高度/灯光同域。这个工具直接从 `/image`
  （usb_cam 原始 JPEG）取帧，**原样落盘、不重新编码**，所以训练用的
  像素和推理时喂给模型的像素是同一个来源。

用法（板子上）:
    source /opt/tros/humble/setup.bash
    python3 tools/capture_server.py --port 8003
    # 浏览器打开 http://<板子IP>:8003/

页面怎么用:
    1. 摆好一类物料（一次只摆一种，数量随意）
    2. 点该类按钮 / 按数字键 → 连拍 N 张
    3. 每拍一组，把物料**转个角度或挪个位置**再拍下一组
    4. 每类凑够 60~100 张（分布在 10~15 组里）

落盘结构:
    <root>/<session>/raw/<LABEL>/<seq>_<ms>.jpg     帧（JPEG 原字节）
    <root>/<session>/raw/unlabeled/<seq>_<ms>.jpg   不分类的帧
    <root>/<session>/meta.jsonl                      每张一行的元数据
    <root>/<session>/session_info.json               会话开始时的相机参数

注意：抓帧期间不要停 det/sort，也不要调 cam_tune 的参数——参数一变，
前后两批帧就不是同一域了（meta.jsonl 里每张都记了当时的参数，可以事后筛）。
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

BOUNDARY = "gcframefeed"

# 与 gc_1029_labels.list 一一对应（模型输出顺序）
LABELS = ["LK", "SMT", "WLZ", "SLZ", "YZHU", "YZ", "BQ", "CUBE"]
CN = {
    "LK": "镂空正方体",
    "SMT": "正四面体",
    "WLZ": "五棱柱",
    "SLZ": "三棱柱",
    "YZHU": "圆柱",
    "YZ": "圆锥",
    "BQ": "半球",
    "CUBE": "正方体",
    "unlabeled": "不分类",
}

CTRLS = [
    ("brightness", "brightness"),
    ("contrast", "contrast"),
    ("saturation", "saturation"),
    ("gamma", "gamma"),
    ("gain", "gain"),
    ("sharpness", "sharpness"),
    ("exposure_time_absolute", "exposure"),
]


# ---------------------------------------------------------------- 帧缓存
class Latest:
    """线程安全的「最新一帧」缓存。seq 保证连拍拿到的是不同帧。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
        self.seq = 0
        self.stamp = 0.0
        self.arrived = 0.0

    def set(self, payload):
        with self.lock:
            self.data = payload
            self.seq += 1
            self.stamp = time.time()
            if not self.arrived:
                self.arrived = self.stamp

    def get(self):
        with self.lock:
            return self.data, self.seq

    def health(self):
        with self.lock:
            age = (time.time() - self.stamp) if self.stamp else -1.0
            span = (self.stamp - self.arrived) if self.arrived else 0.0
            fps = (self.seq / span) if span > 0.5 else 0.0
            return {"frames": self.seq, "age_s": round(age, 2), "fps": round(fps, 2),
                    "bytes": len(self.data) if self.data else 0}


def read_cam_params(device):
    """读 v4l2 当前值；读不到就返回空 dict（不影响抓帧）。

    两种输出格式都兼容：
        brightness: 50                                  （--get-ctrl=名字）
        brightness 0x00980900 (int) : min=.. value=50    （--get-ctrl=all）
    """
    out = {}
    want = set(k for k, _ in CTRLS)
    try:
        # 注意：这块板子的 v4l2-ctl 不支持 --get-ctrl=all（会报
        # unknown control 'all'），只能显式列名字。
        res = subprocess.run(["v4l2-ctl", "-d", device,
                              "--get-ctrl=" + ",".join(sorted(want))],
                             capture_output=True, text=True, timeout=3)
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+0x[0-9A-Fa-f]+.*?:\s*(.*)$", line)
            if m:
                name, rest = m.group(1), m.group(2)
                vm = re.search(r"value=(-?\d+)", rest)
                if vm and name in want:
                    out[name] = int(vm.group(1))
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(-?\d+)\s*$", line)
            if m and m.group(1) in want:
                out[m.group(1)] = int(m.group(2))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- 抓帧任务
class Capturer:
    def __init__(self, latest, root, session, device):
        self.latest = latest
        self.root = os.path.join(root, session)
        self.session = session
        self.device = device
        self.lock = threading.Lock()
        self.seq_no = {}          # cls -> 已用序号
        self.counts = {}          # cls -> 已存张数
        self.queue = []
        self.cv = threading.Condition()
        self.groups = []          # [{"id":.., "cls":.., "files":[...]}]
        self.next_id = 1
        self.started = time.time()
        self.last_error = ""
        self.params = read_cam_params(device)
        self.dims = (0, 0)
        self._load_counts()
        self._write_session_info()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- 初始化 --------------------------------------------------
    def _load_counts(self):
        d = os.path.join(self.root, "raw")
        for cls in LABELS + ["unlabeled"]:
            p = os.path.join(d, cls)
            files = [f for f in os.listdir(p)] if os.path.isdir(p) else []
            self.counts[cls] = len(files)
            seqs = []
            for f in files:
                m = re.match(r"^(\d+)_", f)
                if m:
                    seqs.append(int(m.group(1)))
            self.seq_no[cls] = max(seqs) + 1 if seqs else 1

    def _write_session_info(self):
        os.makedirs(self.root, exist_ok=True)
        info = {
            "session": self.session,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "labels": LABELS,
            "cam_params": self.params,
            "note": "帧为 /image 原始 JPEG 字节，未重新编码",
        }
        with io.open(os.path.join(self.root, "session_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    # -- 对外 ----------------------------------------------------
    def submit(self, cls, n, gap):
        cls = cls.strip()
        if cls not in LABELS and cls != "unlabeled":
            return {"ok": False, "error": "未知类别 %r" % cls}
        try:
            n = max(1, min(60, int(n)))
        except (TypeError, ValueError):
            n = 5
        try:
            gap = max(0.0, min(3.0, float(gap)))
        except (TypeError, ValueError):
            gap = 0.3
        with self.cv:
            job = {"id": self.next_id, "cls": cls, "n": n, "gap": gap}
            self.next_id += 1
            self.queue.append(job)
            self.cv.notify_all()
        return {"ok": True, "job": job}

    def stats(self):
        with self.lock:
            counts = dict(self.counts)
            dims = self.dims
            groups = [g for g in self.groups[-8:]]
        return {
            "session": self.session,
            "root": self.root,
            "total": sum(counts.values()),
            "counts": counts,
            "labels": LABELS,
            "cn": CN,
            "dims": list(dims),
            "params": self.params,
            "pending": len(self.queue),
            "elapsed_s": round(time.time() - self.started, 1),
            "last_error": self.last_error,
            "groups": groups,
            "free_gb": self._free_gb(),
        }

    def _free_gb(self):
        try:
            st = os.statvfs(self.root)
            return round(st.f_bavail * st.f_frsize / 1e9, 1)
        except Exception:
            return -1.0

    def undo(self):
        with self.lock:
            if not self.groups:
                return {"ok": False, "error": "没有可撤销的组"}
            g = self.groups.pop()
            removed = 0
            for p in g["files"]:
                try:
                    os.remove(p)
                    removed += 1
                    cls = g["cls"]
                    self.counts[cls] = max(0, self.counts.get(cls, 1) - 1)
                except OSError:
                    pass
        return {"ok": True, "removed": removed, "cls": g["cls"]}

    # -- 工作线程 ------------------------------------------------
    def _worker(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait(0.5)
                job = self.queue.pop(0)
            try:
                self._run_job(job)
            except Exception as exc:                     # noqa: BLE001
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                sys.stderr.write("[capture] 任务失败 %r\n" % (exc,))

    def _run_job(self, job):
        cls, n, gap = job["cls"], job["n"], job["gap"]
        # 每批重读一次相机参数：参数中途被 cam_tune 改过也能如实记下来
        fresh = read_cam_params(self.device)
        if fresh:
            self.params = fresh
            with self.lock:
                self._write_session_info()
        d = os.path.join(self.root, "raw", cls)
        os.makedirs(d, exist_ok=True)
        files = []
        last_seq = self.latest.get()[1]
        for _ in range(n):
            data, seq = self._wait_new(last_seq, timeout=5.0)
            if data is None:
                self.last_error = "等不到新帧（相机停了吗？）"
                break
            last_seq = seq
            if self.dims == (0, 0):
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with self.lock:
                        self.dims = (img.shape[1], img.shape[0])
            with self.lock:
                no = self.seq_no.get(cls, 1)
                self.seq_no[cls] = no + 1
            name = "%05d_%d.jpg" % (no, int(time.time() * 1000) % 1000000)
            path = os.path.join(d, name)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            files.append(path)
            with self.lock:
                self.counts[cls] = self.counts.get(cls, 0) + 1
            self._meta(path, cls, seq, len(data))
            if gap:
                time.sleep(gap)
        if files:
            with self.lock:
                self.groups.append({"id": job["id"], "cls": cls, "files": files,
                                    "at": time.strftime("%H:%M:%S")})
            sys.stderr.write("[capture] %s +%d 张（累计 %d）\n"
                             % (cls, len(files), self.counts.get(cls, 0)))

    def _wait_new(self, last_seq, timeout):
        end = time.time() + timeout
        while time.time() < end:
            data, seq = self.latest.get()
            if data is not None and seq > last_seq:
                return data, seq
            time.sleep(0.004)
        return None, last_seq

    def _meta(self, path, cls, seq, nbytes):
        rec = {
            "file": os.path.relpath(path, self.root),
            "cls": cls,
            "seq": seq,
            "bytes": nbytes,
            "ts": round(time.time(), 3),
            "params": self.params,
        }
        with io.open(os.path.join(self.root, "meta.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- ROS
class Relay(Node):
    def __init__(self, latest, topic):
        super().__init__("gc_capture_relay")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CompressedImage, topic, self._cb, qos)
        self.latest = latest
        self.get_logger().info("抓帧工具订阅 %s" % topic)

    def _cb(self, msg):
        self.latest.set(bytes(msg.data))


# ---------------------------------------------------------------- 页面
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>GC 抓帧 · 数据集</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#12161b;color:#dfe6ee;font:14px/1.5 system-ui,sans-serif}
#top{display:flex;gap:16px;align-items:center;padding:8px 12px;background:#1b2229;
     border-bottom:1px solid #2c3742;font-family:ui-monospace,Consolas,monospace;font-size:13px;flex-wrap:wrap}
#wrap{display:flex;gap:14px;padding:12px;align-items:flex-start;flex-wrap:wrap}
#left{flex:0 0 auto}
#left img{width:640px;max-width:96vw;display:block;background:#000;border:1px solid #2c3742;border-radius:4px}
#right{flex:1 1 380px;min-width:340px}
#grid{display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));gap:8px}
.cls{position:relative;padding:10px 10px 8px;background:#1d2530;border:1px solid #33414f;
     border-radius:6px;cursor:pointer;text-align:left;color:#dfe6ee;font:inherit;transition:.08s}
.cls:hover{background:#26323f;border-color:#4b6076}
.cls:active{transform:scale(.98)}
.cls b{font-size:15px}
.cls .k{position:absolute;right:8px;top:6px;font-size:11px;color:#7f93a8}
.cls .n{float:right;font-size:18px;font-weight:700;color:#57d38c;font-family:ui-monospace,monospace}
.cls small{color:#8b9bad}
.row{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}
input[type=number]{width:66px;background:#0e1216;color:#dfe6ee;border:1px solid #33414f;
     border-radius:4px;padding:5px 6px;font:inherit}
button.btn{background:#2b6cb0;border:0;color:#fff;padding:7px 12px;border-radius:5px;
     cursor:pointer;font:inherit}
button.btn.gray{background:#3a4756}
button.btn.red{background:#8a3b3b}
#log{height:120px;overflow:auto;background:#0e1216;border:1px solid #2c3742;border-radius:4px;
     padding:6px 8px;font-family:ui-monospace,monospace;font-size:12px;color:#9fb3c8}
.hint{color:#8b9bad;font-size:12.5px;margin:6px 0}
.ok{color:#57d38c}.bad{color:#f18b8b}
</style></head>
<body>
<div id="top">
  <span>抓帧 · 数据集</span>
  <span id="sess">—</span>
  <span id="tot">合计 0</span>
  <span id="fps">fps —</span>
  <span id="dim">—</span>
  <span id="free">—</span>
  <span id="err" class="bad"></span>
</div>
<div id="wrap">
  <div id="left"><img src="/stream" alt="live"></div>
  <div id="right">
    <div class="hint">摆好一类 → 点按钮或按数字键连拍 → <b>转个角度/挪个位置再拍下一组</b></div>
    <div id="grid"></div>
    <div class="row">
      <label>连拍 <input type="number" id="n" value="5" min="1" max="60"> 张</label>
      <label>间隔 <input type="number" id="gap" value="0.3" step="0.1" min="0" max="3"> 秒</label>
      <button class="btn gray" id="undo">撤销上一组</button>
    </div>
    <div class="hint">相机参数（抓帧时逐张记入 meta.jsonl）：<span id="params">—</span></div>
    <div id="log"></div>
    <div class="hint">抓完点这里下载：<a href="/bundle.tar.gz" style="color:#7fb3ff">整包下载 bundle.tar.gz</a>
        · 目录 <span id="root" style="font-family:ui-monospace">—</span></div>
  </div>
</div>
<script>
const LABELS = __LABELS__;
const CN = __CN__;
let SN = {};
function log(t, cls){
  const el = document.getElementById('log');
  const d = document.createElement('div');
  d.className = cls || '';
  d.textContent = new Date().toLocaleTimeString() + '  ' + t;
  el.insertBefore(d, el.firstChild);
  while (el.childNodes.length > 200) el.removeChild(el.lastChild);
}
function build(){
  const g = document.getElementById('grid');
  LABELS.concat(['unlabeled']).forEach((L, i) => {
    const b = document.createElement('button');
    b.className = 'cls'; b.id = 'b_' + L;
    b.innerHTML = '<span class="k">' + (i < 8 ? (i + 1) : '0') + '</span>' +
                  '<b>' + CN[L] + '</b> <small>' + L + '</small>' +
                  '<span class="n" id="n_' + L + '">0</span>';
    b.onclick = () => cap(L);
    g.appendChild(b);
  });
}
function cap(L){
  const n = document.getElementById('n').value || 5;
  const gap = document.getElementById('gap').value || 0.3;
  const t0 = performance.now();
  fetch('/capture?cls=' + L + '&n=' + n + '&gap=' + gap)
    .then(r => r.json())
    .then(j => {
      if (j.ok) log('已排入 ' + CN[L] + ' × ' + j.job.n + ' 张', 'ok');
      else log('失败：' + j.error, 'bad');
    })
    .catch(e => log('请求失败 ' + e, 'bad'));
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key >= '1' && e.key <= '8') cap(LABELS[parseInt(e.key, 10) - 1]);
  if (e.key === '0') cap('unlabeled');
});
document.getElementById('undo').onclick = () => {
  fetch('/undo', {method: 'POST'}).then(r => r.json()).then(j => {
    j.ok ? log('已撤销 ' + CN[j.cls] + ' 一组，删了 ' + j.removed + ' 张', 'ok')
         : log('撤销失败：' + j.error, 'bad');
  });
};
function tick(){
  fetch('/stats').then(r => r.json()).then(s => {
    document.getElementById('sess').textContent = '会话 ' + s.session;
    document.getElementById('tot').textContent = '合计 ' + s.total + ' 张';
    document.getElementById('root').textContent = s.root;
    document.getElementById('free').textContent = '剩余 ' + s.free_gb + ' GB';
    const c = s.counts || {};
    LABELS.concat(['unlabeled']).forEach(L => {
      const el = document.getElementById('n_' + L);
      if (el) el.textContent = c[L] || 0;
    });
    const h = document.getElementById('params');
    if (s.params && Object.keys(s.params).length)
      h.textContent = Object.entries(s.params).map(([k, v]) => k + '=' + v).join('  ');
    if (s.last_error) document.getElementById('err').textContent = '错误：' + s.last_error;
  }).catch(() => {});
  fetch('/healthz').then(r => r.json()).then(h => {
    document.getElementById('fps').textContent = 'fps ' + h.fps;
    document.getElementById('dim').textContent = h.bytes ? (Math.round(h.bytes / 1024) + ' KB/帧') : '无画面';
  }).catch(() => {});
}
build(); tick(); setInterval(tick, 1500);
</script>
</body></html>"""


def make_handler(latest, cap):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.startswith("/undo"):
                self._send(200, cap.undo())
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_GET(self):
            path, _, query = self.path.partition("?")
            qs = {}
            for kv in query.split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    qs[k] = v
            if path in ("/", "/index.html"):
                body = (PAGE.replace("__LABELS__", json.dumps(LABELS))
                            .replace("__CN__", json.dumps(CN, ensure_ascii=False)))
                self._send(200, body, "text/html; charset=utf-8")
            elif path == "/healthz":
                self._send(200, latest.health())
            elif path == "/stats":
                self._send(200, cap.stats())
            elif path == "/capture":
                cls = qs.get("cls", "unlabeled")
                n = qs.get("n", 5)
                gap = qs.get("gap", 0.3)
                self._send(200, cap.submit(cls, n, gap))
            elif path == "/last.jpg":
                data, _ = latest.get()
                if not data:
                    self._send(503, b"no frame", "text/plain")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while True:
                        data, _ = latest.get()
                        if data is not None and data is not last:
                            last = data
                            self.wfile.write(("--%s\r\nContent-Type: image/jpeg\r\n"
                                              "Content-Length: %d\r\n\r\n"
                                              % (BOUNDARY, len(data))).encode())
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        else:
                            time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif path == "/bundle.tar.gz":
                self._send_bundle()
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def _send_bundle(self):
            tmp = os.path.join(cap.root, "_bundle.tar.gz")
            try:
                with tarfile.open(tmp, "w:gz") as tf:
                    tf.add(cap.root, arcname=os.path.basename(cap.root))
                size = os.path.getsize(tmp)
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Disposition",
                                 'attachment; filename="%s.tar.gz"' % os.path.basename(cap.root))
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with open(tmp, "rb") as f:
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="抓帧工具（板端网页版）")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--topic", default="/image")
    ap.add_argument("--root", default="/root/vision_ws/dataset")
    ap.add_argument("--session", default="", help="会话名，默认按时间生成")
    ap.add_argument("--device", default="/dev/cam0")
    args = ap.parse_args()

    session = args.session.strip() or time.strftime("%Y%m%d_%H%M")
    rclpy.init()
    latest = Latest()
    node = Relay(latest, args.topic)
    cap = Capturer(latest, args.root, session, args.device)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(latest, cap))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    node.get_logger().info("抓帧页 http://0.0.0.0:%d/  会话=%s  目录=%s"
                           % (args.port, session, cap.root))
    print("抓帧页: http://<板子IP>:%d/   会话 %s   目录 %s" % (args.port, session, cap.root))

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()