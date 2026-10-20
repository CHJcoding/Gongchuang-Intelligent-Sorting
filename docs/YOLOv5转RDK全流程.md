# YOLOv5 → RDK X5 部署全流程（对应你的 best.pt，v5 6.2，7类）

> 你的模型：`C:\Users\34559\Desktop\GC\best.pt`（myYOLOv5s，7 类：BQ/LK/SLZ/SMT/WLZ/YZ/YZHU）
> 注意：这是 **YOLOv5** 流程，跟你那份《模型转换全流程（YOLOv8→RDK X5）.txt》不一样，别照抄 v8 的改法。

---

## 环境一：Windows PC（改 head + 导 ONNX）

### 1. 备份原始文件
```powershell
copy D:\yolov5\models\yolo.py D:\yolov5\models\yolo.py.bak
```

### 2. 修改 Detect.forward（导出 raw 输出，不解码）

打开 `D:\yolov5\models\yolo.py`，找到 class Detect 的 forward 方法（约第 91 行），
在 `if not self.training:` 分支里，**export 模式下直接返回 raw conv 输出**（不做 sigmoid/grid 解码，
因为 BPU 上后处理库自己解）。

把 forward 方法整体改成：

```python
def forward(self, x):
    z = []
    for i in range(self.nl):
        x[i] = self.m[i](x[i])          # conv: (bs, na*no, ny, nx)
        bs, _, ny, nx = x[i].shape

        if self.export:
            # RDK 部署：直接输出 raw feature map，不做任何解码
            # 形状 (bs, na*no, ny, nx) = (bs, 3*(7+5), ny, nx) = (bs, 36, ny, nx)
            z.append(x[i])
            continue

        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
        xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
        xy = (xy * 2 + self.grid[i]) * self.stride[i]
        wh = (wh * 2) ** 2 * self.anchor_grid[i]
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, self.na * nx * ny, self.no))

    if self.training:
        return x
    elif self.export:
        # 返回 3 个尺度的 raw feature map（BPU 后处理用）
        return tuple(z)
    else:
        return (torch.cat(z, 1),)
```

### 3. 导出 ONNX（opset=11）

在 PowerShell 里：
```powershell
D:\yolo\envs\yolo\python.exe -c "
import sys, inspect, torch
sys.path.insert(0, r'D:\yolov5')
_orig = torch.load
def _compat(*a, **k):
    k.pop('weights_only', None)
    k['weights_only'] = False
    return _orig(*a, **k)
torch.load = _compat

from models.experimental import attempt_load
m = attempt_load(r'C:\Users\34559\Desktop\GC\best.pt', device='cpu')
m.eval()

import torch.onnx
dummy = torch.zeros(1, 3, 640, 640)
# 关键：把 export 标志打开
m.model[-1].export = True if hasattr(m.model[-1], 'export') else None
# 对于 attempt_load 加载的 DetectionModel，最后一层就是 Detect
detect_layer = m.model[-1]
detect_layer.export = True

torch.onnx.export(
    detect_layer and m,        # 整个模型
    dummy,
    r'C:\Users\34559\Desktop\GC\best_v5.onnx',
    opset_version=11,
    input_names=['images'],
    output_names=['output_0', 'output_1', 'output_2'],  # 3 个尺度
    dynamic_axes=None,
)
print('ONNX exported')
"
```

导出后检查：
```powershell
# 用 onnx 库检查输入输出形状
D:\yolo\envs\yolo\python.exe -c "
import onnx
m = onnx.load(r'C:\Users\34559\Desktop\GC\best_v5.onnx')
print('inputs:', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print('outputs:', [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
"
```
应该看到：
- input: `images` (1, 3, 640, 640)
- outputs: 3 个，形状分别是 (1, 36, 80, 80)、(1, 36, 40, 40)、(1, 36, 20, 20)

---

## 环境二：Ubuntu 虚拟机（Docker 量化转 .bin）

### 1. 建目录
```bash
mkdir -p ~/Desktop/x5_project/{onnx,config,output,calib_data/images}
```

### 2. 把 Windows 上的 best_v5.onnx 传进来
用 scp / U盘 / 共享文件夹，放到 `~/Desktop/x5_project/onnx/best.onnx`

### 3. 校准图片
从你 PC 的 `C:\Users\34559\Desktop\GC\mydata\images\val\` 里挑 100 张真实场景图，
复制到 `~/Desktop/x5_project/calib_data/images/`（不用标注图，原图就行）。

### 4. 启动 Docker 容器
```bash
docker run -it --name x5_dev \
  -v ~/Desktop/x5_project:/workspace \
  -w /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310
```

### 5. 写转换配置 config/convert.yaml

```yaml
model_parameters:
  onnx_model: '/workspace/onnx/best.onnx'
  march: 'bayes-e'
  working_dir: '/workspace/output'
  output_model_file_prefix: 'yolov5_best'

input_parameters:
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451

calibration_parameters:
  cal_data_dir: '/workspace/calib_data/images'
  cal_data_type: 'float32'
  preprocess_on: True
```

### 6. 执行转换
```bash
hb_mapper makertbin --model-type onnx --config config/convert.yaml
```
成功后 `/workspace/output/` 下会生成 `yolov5_best.bin`。

### 7. 退出容器，取文件
```bash
exit
ls ~/Desktop/x5_project/output/
```

---

## 环境三：RDK X5（板上部署）

### 1. scp 传 bin 到板上
在 Ubuntu 虚拟机里：
```bash
scp ~/Desktop/x5_project/output/yolov5_best.bin root@<板IP>:/root/
# 例如：scp ~/Desktop/x5_project/output/yolov5_best.bin root@10.14.143.122:/root/
```

### 2. 板上装推理库
```bash
pip install bpu_infer_lib_x5 -i http://archive.d-robotics.cc/simple/ --trusted-host archive.d-robotics.cc
```

### 3. 拉官方 YOLOv5 示例（不是 v8！）
```bash
cd ~
git clone https://github.com/D-Robotics/rdk_model_zoo.git
# 找 yolov5 示例（不是 ultralytics_yolo 那个 v8）
ls rdk_model_zoo/samples/vision/
```

### 4. 配置文件改模型路径
找到 v5 示例的 config.json，把 model_path 改成 `/root/yolov5_best.bin`，
确认类别数改成 7、anchors 跟你训练时一致（v5s 默认 anchors 要核对 best.pt 里的）。

### 5. 启动推理
```bash
source /opt/tros/humble/setup.bash
export CAM_TYPE=usb
ros2 launch <v5示例包名> <launch文件>.launch.py
```

---

## 常见坑

| 问题 | 原因 | 解决 |
|---|---|---|
| ONNX 导出报错 weights_only | torch 2.6+ 拒绝老权重 | 前面脚本已包装 torch.load |
| 转换时说 output 数量不对 | head 改法不对，没输出 3 个尺度 | 检查 forward 里 export 分支 |
| 板上检测全是空框 | anchors 跟训练时不一致 | 打开 best.pt 看 anchors，写进 config.json |
| 检测类别错 | nc 配置错 | 改成 7，顺序跟 names 一致 |
| fps 很低 | 没开 BPU，跑在 CPU | 确认 bpu_infer_lib 装好，用 tros launch |
