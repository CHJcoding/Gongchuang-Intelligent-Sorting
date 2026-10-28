#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把目录里最新的 JPEG 转成网页 MJPEG —— 纯标准库，不依赖 ROS / opencv。

干什么用：业务节点（sort_vision_node.py）在 `config.ANNOTATE_ENABLE = True` 时
每 1/ANNOTATE_FPS 秒往 `ANNOTATE_DIR` 写一张 `last.jpg`（里面画着**贴合旋转框 +
抓取点**）。本脚本把它变成浏览器能看的流：

    python3 tools/mjpeg_view.py                      # 默认看 /root/vision_ws/preview
    python3 tools/mjpeg_view.py --dir ./preview --port 8081
    # 浏览器打开 http://<板子IP>:8001

为什么单独一个进程：
  * 节点里再开 socket 线程，写错一行就把分拣节点带崩；
  * 这个进程挂了/重启了，分拣照常跑（它只是读文件）；
  * 想在电脑上看，不需要装 ROS，也不需要 X11 转发。

接口：
  GET /            只看图（自动刷新，每 200ms 换一张）
  GET /stream      MJPEG 流（multipart/x-mixed-replace，VLC / img 标签可用）
  GET /last.jpg    当前最新一张原图
  GET /healthz     返回 {"ok":true,"age_s":..,"n_frames":..} 方便脚本探测
"""

import argparse
import glob
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOUNDARY = "gcjpegframe"

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>抓取点预览 %(name)s</title>
<style>
 body{background:#111;color:#ddd;font-family:monospace;margin:0;padding:8px}
 img{max-width:100%%;border:1px solid #444}
 .bar{padding:4px 0;font-size:13px}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%%;
      background:#0c0;margin-right:6px;vertical-align:middle}
</style>
<script>
function tick(){
  var im=document.getElementById('v');
  im.src='/last.jpg?t='+Date.now();
}
setInterval(tick, 200);
</script>
</head><body>
<div class="bar"><span class="dot"></span>%(name)s &nbsp; 黄=贴合旋转框 &nbsp;
青=抓取点 &nbsp; 红虚线=回退(用框中心)</div>
<div class="bar">目录 %(dir)s &nbsp;|&nbsp; <span id="st">等第一张图…</span></div>
<img id="v" src="/last.jpg">
<script>
setInterval(function(){
 fetch('/healthz').then(r=>r.json()).then(function(j){
   document.getElementById('st').textContent =
     '第 '+j.n_frames+' 张 / 距今 '+j.age_s.toFixed(1)+'s';
 }).catch(function(){
   document.getElementById('st').textContent='读不到状态';
 });
}, 1000);
</script>
</body></html>
"""


class LatestImage(object):
    """盯住一个目录，永远给出最新的 jpg（按 mtime）。"""

    def __init__(self, directory, pattern="*.jpg", fresh_s=5.0):
        self.dir = directory
        self.pattern = pattern
        self.fresh_s = fresh_s
        self.n_frames = 0
        self._last_path = None
        self._last_mtime = 0.0
        self._lock = threading.Lock()

    def newest(self):
        """返回 (path, mtime) 或 (None, 0)。"""
        try:
            cands = glob.glob(os.path.join(self.dir, self.pattern))
        except Exception:
            return None, 0.0
        best, best_m = None, 0.0
        for p in cands:
            try:
                m = os.stat(p).st_mtime
            except OSError:
                continue
            if m > best_m:
                best, best_m = p, m
        return best, best_m

    def read(self):
        """返回最新一张的字节；没有则 None。"""
        with self._lock:
            path, mtime = self.newest()
            if path is None:
                return None
            if path != self._last_path or mtime != self._last_mtime:
                self._last_path, self._last_mtime = path, mtime
                self.n_frames += 1
            try:
                with io.open(path, "rb") as f:
                    return f.read()
            except OSError:
                return None

    def age(self):
        _p, mtime = self.newest()
        return 0.0 if not mtime else max(0.0, time.time() - mtime)

    def healthy(self):
        return self.age() <= self.fresh_s


def make_handler(latest, name="preview"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):      # 静音，避免刷屏
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                body = (_PAGE % {"name": name, "dir": latest.dir}).encode("utf-8")
                self._send(200, "text/html; charset=utf-8", body)
                return
            if path == "/healthz":
                body = json.dumps({"ok": latest.healthy(),
                                   "age_s": latest.age(),
                                   "n_frames": latest.n_frames,
                                   "dir": latest.dir}).encode()
                self._send(200, "application/json", body)
                return
            if path == "/last.jpg":
                data = latest.read()
                if data is None:
                    self._send(404, "text/plain", b"no image yet")
                else:
                    self._send(200, "image/jpeg", data)
                return
            if path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=%s" % BOUNDARY)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while True:
                        data = latest.read()
                        if data is not None and data != last:
                            last = data
                            self.wfile.write(
                                ("--%s\r\nContent-Type: image/jpeg\r\n"
                                 "Content-Length: %d\r\n\r\n"
                                 % (BOUNDARY, len(data))).encode())
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                return
            self._send(404, "text/plain", b"not found")
    return Handler


def serve(directory, port=8001, host="0.0.0.0", name=None, fresh_s=5.0):
    latest = LatestImage(directory, fresh_s=fresh_s)
    latest.read()          # 预热，别让 n_frames 从 0 开始
    httpd = ThreadingHTTPServer((host, port), make_handler(latest, name or "preview"))
    httpd.daemon_threads = True
    return httpd, latest


def _selftest():
    """本机自检：造一张图 → 起服务 → 用 urllib 把三个接口都打一遍。"""
    import tempfile
    import urllib.request

    tmp = tempfile.mkdtemp(prefix="mjpeg_selftest_")
    p = os.path.join(tmp, "last.jpg")
    with io.open(p, "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")          # 最小合法 JPEG 头尾

    httpd, latest = serve(tmp, port=0, host="127.0.0.1")
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = "http://127.0.0.1:%d" % port
    ok = True
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            html = r.read().decode("utf-8")
            assert r.status == 200 and "抓取点预览" in html, "页面不对"
        with urllib.request.urlopen(base + "/last.jpg", timeout=5) as r:
            assert r.status == 200 and r.read().startswith(b"\xff\xd8"), "图不对"
        with urllib.request.urlopen(base + "/healthz", timeout=5) as r:
            j = json.loads(r.read().decode("utf-8"))
            assert j["ok"] is True and j["n_frames"] >= 1, j
        # 流接口：读头就够（不读完，否则会一直挂）
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", "/stream")
        resp = c.getresponse()
        assert resp.status == 200, resp.status
        assert "multipart/x-mixed-replace" in resp.getheader("Content-Type", "")
        c.close()
        print("[selftest] 三个接口均正常 (port=%d, frames=%d)" % (port, latest.n_frames))
    except AssertionError as e:
        ok = False
        print("[selftest] 失败:", e)
    finally:
        httpd.shutdown()
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="把最新 JPEG 转成网页 MJPEG（抓取点预览）")
    ap.add_argument("--dir", default="/root/vision_ws/preview",
                    help="节点写图的目录（config.ANNOTATE_DIR）")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--name", default=None, help="页面标题")
    ap.add_argument("--fresh", type=float, default=5.0,
                    help="图超过这么多秒没更新就报不健康（默认 5s）")
    ap.add_argument("--selftest", action="store_true", help="本机自检后退出")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if _selftest() else 1

    if not os.path.isdir(args.dir):
        print("目录不存在: %s（节点还没写过图？先确认 ANNOTATE_ENABLE=True）"
              % args.dir)
    httpd, _latest = serve(args.dir, args.port, args.host, args.name, args.fresh)
    print("抓取点预览: http://<板子IP>:%d   (目录 %s)" % (args.port, args.dir))
    print("  /last.jpg 当前图   /stream MJPEG   /healthz 健康检查")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())