#!/usr/bin/env python3
import sys, cv2
from pathlib import Path

# 加路径
sys.path.insert(0, '/root/rdk_model_zoo')

from samples.vision.yolov5.runtime.python.cli import ModelSelection, ANCHORS, STRIDES
from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
from utils.py_utils.assets import Asset

# 构造假 Asset（绕过官方校验）
fake_asset = Asset(
    group='x5',
    sample_id='yolov5',
    filename='yolov5_best.bin',
    format='bin',
    url=None,
    sha256=None,
)

# 构造 ModelSelection，model_path 指向你的 .bin
selection = ModelSelection(
    asset=fake_asset,
    target='x5',
    model_path=Path('/root/vision_ws/yolov5_best.bin'),
    variant='s-v7.0',
    consumer='yolov5',
)

# 类别名
labels = ['BQ','LK','SLZ','SMT','WLZ','YZ','YZHU']

# 创建推理任务
task = YOLOv5Task(
    selection,
    score_thres=0.25,
    nms_thres=0.45,
    anchors=ANCHORS,
)
task.set_scheduling_params(priority=0, bpu_cores=(0,1))

# 读测试图
img_path = sys.argv[1] if len(sys.argv) > 1 else '/root/rdk_model_zoo/samples/vision/yolov5/test_data/bus.jpg'
image = cv2.imread(img_path)
if image is None:
    print(f'Cannot read {img_path}')
    sys.exit(1)

print(f'Input: {image.shape}')
result = task.predict(image, resize_type=0)

# 画结果
for det in result:
    x1,y1,x2,y2, conf, cls = det
    name = labels[int(cls)] if int(cls) < len(labels) else str(int(cls))
    cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.putText(image, f'{name} {conf:.2f}', (int(x1),int(y1)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    print(f'{name} {conf:.2f} at ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})')

cv2.imwrite('/root/vision_ws/result.jpg', image)
print('Saved /root/vision_ws/result.jpg')
