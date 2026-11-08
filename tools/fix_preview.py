# -*- coding: utf-8 -*-
import time, shutil

def bak(p):
    shutil.copy2(p, p + ".bak." + time.strftime("%m%d_%H%M"))

p = "/root/vision_ws/vision_sorter/config.py"
s = open(p, encoding="utf-8").read()
old = "DEBUG_IMAGE_JPEG_QUALITY = 80"
assert s.count(old) == 1
bak(p)
s = s.replace(old, "DEBUG_IMAGE_JPEG_QUALITY = 92   # 80->92：预览要看清细节，80 在 640x480 上已经能看出糊")
open(p, "w", encoding="utf-8").write(s)
print("config: JPEG 质量 80 -> 92")

p = "/root/vision_ws/tools/preview_server.py"
s = open(p, encoding="utf-8").read()
old = "img{width:100%;max-width:960px;display:block;margin:0 auto}"
assert s.count(old) == 1
bak(p)
new = "img{width:100%;max-width:640px;display:block;margin:0 auto;image-rendering:auto}"
s = s.replace(old, new)
s = s.replace('GC 识别预览 · /image_annotated', 'GC 识别预览 · /image_annotated · 原生 640x480（1:1 不放大）')
open(p, "w", encoding="utf-8").write(s)
print("preview: max-width 960 -> 640（停止浏览器放大）")
