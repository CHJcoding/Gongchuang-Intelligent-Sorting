# -*- coding: utf-8 -*-
"""货物清单配置：货物数据类 + 默认货物清单（8 种几何体/物块）+ 占位图自动生成

货物图片路径为空或文件不存在时，自动生成纯色占位图（Pillow），
存放到项目 assets/goods_placeholders 目录，供统计表格图片列兜底展示。
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER_DIR = os.path.join(PROJECT_ROOT, "assets", "goods_placeholders")
PLACEHOLDER_SIZE = (120, 120)  # 占位图尺寸 (宽, 高)

# color_key -> RGB 纯色（用于占位图；与 ColorDetector 的颜色标识对齐）
COLOR_KEY_RGB: Dict[str, tuple] = {
    "yellow": (255, 214, 40),
    "red":    (226, 51, 51),
    "green":  (56, 173, 75),
    "blue":   (0, 0, 255),
    "white":  (235, 235, 235),
}


@dataclass
class GoodsItem:
    """货物数据类

    Attributes:
        goods_id: 货物唯一 ID（如 G001）
        name: 货物名称（表格展示用）
        color_key: 对应颜色标识（与 ColorDetector 的 color_key 对齐：yellow/red/green/blue/white）
        image_path: 图片路径；为空或文件不存在时自动生成/复用纯色占位图
        thresholds: 可选，覆盖该颜色的 HSV 阈值（None 时使用 ColorDetector 默认阈值）
    """
    goods_id: str
    name: str
    color_key: str
    image_path: str = ""
    thresholds: Optional[Dict[str, int]] = None


def get_default_goods() -> List[GoodsItem]:
    """默认货物清单（7 种几何体；颜色识别后续采用 YOLO+HSV，
    几何体 color_key 暂留空预留）"""
    goods_dir = os.path.join(PROJECT_ROOT, "assets", "goods")
    return [
        GoodsItem("G001", "镂空正方体", "", os.path.join(goods_dir, "goods_01_hollow_cube.png")),
        GoodsItem("G002", "正四面体", "", os.path.join(goods_dir, "goods_02_tetrahedron.png")),
        GoodsItem("G003", "五棱柱", "", os.path.join(goods_dir, "goods_03_pentagonal_prism.png")),
        GoodsItem("G004", "三棱柱", "", os.path.join(goods_dir, "goods_04_triangular_prism.png")),
        GoodsItem("G005", "圆锥", "", os.path.join(goods_dir, "goods_05_cone.png")),
        GoodsItem("G006", "圆柱", "", os.path.join(goods_dir, "goods_06_cylinder.png")),
        GoodsItem("G007", "半球", "", os.path.join(goods_dir, "goods_07_hemisphere.png")),
    ]


def get_image_path(item: GoodsItem) -> str:
    """返回货物图片路径；未配置或文件不存在时返回占位图路径（并确保已生成）"""
    if item.image_path and os.path.isfile(item.image_path):
        return item.image_path
    return ensure_placeholder(item)


def ensure_placeholder(item: GoodsItem, save_dir: Optional[str] = None) -> str:
    """为货物生成纯色占位图（已存在则复用），返回 PNG 绝对路径"""
    save_dir = save_dir or PLACEHOLDER_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{item.goods_id}_{item.color_key}.png")
    if os.path.isfile(path):
        return path
    rgb = COLOR_KEY_RGB.get(item.color_key, (180, 180, 180))
    try:
        from PIL import Image
        img = Image.new("RGB", PLACEHOLDER_SIZE, rgb)
        img.save(path, "PNG")
    except Exception:
        # Pillow 不可用时写最小合法 PNG（纯色），保证界面不崩
        _write_minimal_png(path, rgb)
    return path


def _write_minimal_png(path: str, rgb: tuple) -> None:
    """写 1x1 纯色最小合法 PNG（无压缩），不依赖 Pillow"""
    import struct
    import zlib

    w = h = 1
    raw = b"\x00" + bytes(rgb)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
