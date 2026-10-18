# -*- coding: utf-8 -*-
"""六色色块检测核心（纯 OpenCV）

识别黄/红/绿/蓝/白/黑六种色块（智能车工训识别物，约 10-15cm），检测流程：
GaussianBlur -> HSV -> 六色掩码 -> 开运算 -> findContours -> 面积/宽高比筛选 -> 输出

阈值要点（可行性评估报告实测初始值）：
- 黄：H 15~35，S≥80，V≥100
- 红：双区间合并 (H 0~16) ∪ (H 168~180)，S≥80，V≥90
- 绿：H 55~85，S≥80，V≥75
- 蓝：H 86~135，S≥120，V≥60（标准蓝 BGR(255,0,0) 对应 HSV H≈120）
- 白：高 V 低 S（V≥200 且 S≤60），不看色相；高光点可能被误检需靠面积筛选
- 黑：低饱和低亮（S≤45 且 V≤45），不看色相；紧致度过滤避免深色/暗面误判

支持 ROI 限定：detect(frame, roi=...) 时仅检测框选区域内色块，框外不产生检测框。

只输出颜色名 + 检测框 + 中心点（供显示与后续坐标输出预留），本模块不做坐标映射。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 颜色元信息：color_key -> (中文名, 框线 BGR 颜色)
COLOR_META: Dict[str, Tuple[str, Tuple[int, int, int]]] = {
    "yellow": ("yellow", (0, 255, 255)),
    "red":    ("red", (0, 0, 255)),
    "green":  ("green", (0, 255, 0)),
    "blue":   ("blue", (255, 0, 0)),
    "black":  ("black", (180, 180, 180)),  # 灰框线，与白色品红框区分
    "white":  ("white", (255, 0, 255)),  # 品红框线，与黑色灰框区分
}

# 可行性评估报告实测初始阈值
DEFAULT_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "yellow": {"h_low": 15, "h_high": 35, "s_low": 80, "v_low": 100},
    "red":    {"h_low": 0, "h_high": 16, "h_low2": 168, "h_high2": 180,
               "s_low": 80, "v_low": 90},
    "green":  {"h_low": 55, "h_high": 85, "s_low": 80, "v_low": 75},
    "blue":   {"h_low": 86, "h_high": 135, "s_low": 120, "v_low": 60},
    "black":  {"s_high": 45, "v_high": 45},  # 黑：低饱和 + 低亮度，不看色相
    "white":  {"v_low": 200, "s_high": 60},
}


def _merge_thresholds(custom: Optional[Dict]) -> Dict[str, Dict[str, int]]:
    """将用户自定义阈值与默认值合并（只覆盖传入项）"""
    merged: Dict[str, Dict[str, int]] = {}
    for key, base in DEFAULT_THRESHOLDS.items():
        d = dict(base)
        if custom and key in custom:
            d.update(custom[key])
        merged[key] = d
    return merged


@dataclass
class ColorDetection:
    """单个色块检测结果"""

    color_key: str                        # yellow / red / green / blue / black / white
    color_name: str                       # 中文名：黄 / 红 / 绿 / 蓝 / 黑 / 白
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) 原始帧像素
    center: Tuple[float, float]           # (cx, cy) 中心点
    area: float                           # 轮廓面积（像素）


class ColorDetector:
    """六色色块检测器（黄/红/绿/蓝/白/黑，纯 OpenCV，无模型依赖）"""

    def __init__(self, thresholds: Optional[Dict] = None,
                 min_area: int = 400,
                 max_area_ratio: float = 0.9,
                 min_wh_ratio: float = 0.25,
                 max_wh_ratio: float = 4.0,
                 black_compactness_min: float = 0.35):
        # 默认遍历顺序：黄、红、绿、蓝、白、黑
        self.thresholds = _merge_thresholds(thresholds)
        self.min_area = min_area                # 轮廓最小面积（像素）
        self.max_area_ratio = max_area_ratio    # 占整帧面积上限，过滤整幅同色
        self.min_wh_ratio = min_wh_ratio        # 宽高比下限（排除细长线状）
        self.max_wh_ratio = max_wh_ratio        # 宽高比上限
        self.black_compactness_min = black_compactness_min  # 黑色通道紧致度下限（滤深色/暗面散块）

    # ---------- 对外接口 ----------
    def detect(self, frame, roi=None) -> List[ColorDetection]:
        """对一帧执行六色检测（黄/红/绿/蓝/白/黑），返回检测结果列表（颜色名 + 框 + 中心点）

        Args:
            frame: 原始 BGR 帧
            roi: 可选 (x1, y1, x2, y2) 识别范围（原始帧坐标）。
                 传入时仅在该范围内检测，框外区域不产生任何检测框；
                 None 表示全画面检测。
        """
        if frame is None:
            return []
        h, w = frame.shape[:2]

        # ROI 限定：裁剪子图检测，检测结果坐标回加偏移还原到原始帧坐标系
        if roi is not None:
            x1, y1, x2, y2 = [int(v) for v in roi]
            x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                return []
            dets = self._detect_sub(frame[y1:y2, x1:x2])
            for d in dets:
                d.box = (d.box[0] + x1, d.box[1] + y1,
                         d.box[2] + x1, d.box[3] + y1)
                d.center = (d.center[0] + x1, d.center[1] + y1)
            return dets
        return self._detect_sub(frame)

    def _detect_sub(self, sub) -> List[ColorDetection]:
        """对单帧或 ROI 子图执行六色检测，返回子图坐标系下的结果"""
        if sub is None or sub.size == 0:
            return []
        h, w = sub.shape[:2]
        frame_area = float(h * w)

        blurred = cv2.GaussianBlur(sub, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        dets: List[ColorDetection] = []
        # 遍历顺序：黄、红、绿、蓝、白、黑（黑放最后，阈值低饱和低亮不与彩色通道冲突）
        for key in ("yellow", "red", "green", "blue", "white", "black"):
            mask = self._build_mask(hsv, self.thresholds[key])
            # 形态学去噪：开运算去小噪点，闭运算补内部孔洞
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            name = COLOR_META[key][0]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > self.max_area_ratio * frame_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw <= 0 or bh <= 0:
                    continue
                # 黑色通道额外紧致度过滤：深色/暗面常呈散块或条状，紧致度低，
                # 实心黑色物块紧致度接近 1（圆≈0.785、方≈1），低于下限则判为误报跳过
                if key == "black" and self.black_compactness_min > 0:
                    rect_area = float(bw * bh)
                    if rect_area > 0 and area / rect_area < self.black_compactness_min:
                        continue
                wh = bw / float(bh)
                if wh < self.min_wh_ratio or wh > self.max_wh_ratio:
                    continue
                box = (float(x), float(y), float(x + bw), float(y + bh))
                center = (float(x + bw / 2.0), float(y + bh / 2.0))
                dets.append(ColorDetection(key, name, box, center, float(area)))
        return dets

    def classify_roi(self, frame, box, min_hit_ratio: float = 0.10):
        """对检测框内中心区域做主色判定（HSV 阈值投票）

        只统计框中心约 60% 矩形区域的像素：用现有默认阈值逐色生成掩码，
        命中像素占比最高者胜出；最高占比低于 min_hit_ratio 时返回 None
        （说明框内无六色主色，颜色未判出）。

        Args:
            frame: 原始 BGR 帧
            box: (x1, y1, x2, y2) 检测框（原始帧坐标）
            min_hit_ratio: 命中像素占比下限（0~1），低于该值判定失败

        Returns:
            主色 color_key（yellow/red/green/blue/white/black）；无法判定时 None
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        # 中心 60% 矩形区域（各向内收缩 20%）；框过小时退回全框
        cx1 = x1 + int(bw * 0.2)
        cy1 = y1 + int(bh * 0.2)
        cx2 = x2 - int(bw * 0.2)
        cy2 = y2 - int(bh * 0.2)
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2
        roi = frame[cy1:cy2, cx1:cx2]
        total = float(roi.shape[0] * roi.shape[1])
        if total <= 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        best_key = None
        best_ratio = 0.0
        # 遍历顺序与 detect 保持一致：黄、红、绿、蓝、白、黑
        for key in ("yellow", "red", "green", "blue", "white", "black"):
            mask = self._build_mask(hsv, self.thresholds[key])
            ratio = float(cv2.countNonZero(mask)) / total
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
        if best_ratio < min_hit_ratio:
            return None
        return best_key

    def update_threshold(self, color_key: str, **params) -> None:
        """在线微调某个颜色的 HSV 阈值参数"""
        if color_key not in self.thresholds:
            return
        self.thresholds[color_key].update(params)

    def reset_thresholds(self) -> None:
        """恢复默认阈值"""
        self.thresholds = _merge_thresholds(None)

    # ---------- 内部 ----------
    @staticmethod
    def _build_mask(hsv, t: Dict[str, int]) -> np.ndarray:
        """根据颜色阈值生成单色掩码；黑看 V 上限（可叠加 S 上限），白看 V 下限+S 上限，红做双区间合并"""
        # 黑色：不看色相，低饱和 + 低亮度（V 上限）；默认不设 S 上限，兼容旧版 v_high 用法
        if "v_high" in t:
            s_hi = t.get("s_high", 255)
            return cv2.inRange(hsv, (0, 0, 0), (180, s_hi, t["v_high"]))
        # 白色/灰白：不看色相，V 下限 + S 上限（高 V 低 S）
        if "v_low" in t and "s_high" in t:
            return cv2.inRange(hsv, (0, 0, t["v_low"]), (180, t["s_high"], 255))

        s_lo = t.get("s_low", 0)
        v_lo = t.get("v_low", 0)
        mask = cv2.inRange(hsv, (t["h_low"], s_lo, v_lo),
                           (t["h_high"], 255, 255))
        # 红色双区间合并（H 两端）
        if "h_low2" in t:
            mask2 = cv2.inRange(hsv, (t["h_low2"], s_lo, v_lo),
                                (t["h_high2"], 255, 255))
            mask = cv2.bitwise_or(mask, mask2)
        return mask
