# -*- coding: utf-8 -*-
"""分拣状态机与计数逻辑（纯 Python 回调，不依赖 PyQt）

状态流转（机械臂动作完成 + 复位后视觉校验通过才 +1）：
    空闲(IDLE) → 已锁定(LOCKED) → 已下发(DISPATCHED) → 等待动作完成(WAIT_ACTION)
    → 等待复位(WAIT_RESET) → 校验(VERIFY) → 成功(SUCCESS, 该货物计数+1) / 失败(FAIL, 不计数)
    → 回空闲(IDLE)

校验逻辑：锁定目标中心坐标容差圈（半径可配置，默认 50px）内若仍存在同色货物中心点，
视为分拣失败（货物未被移走）；无则视为成功（货物已移走）。

对外回调（通过 on_state_changed / on_count_changed / on_result_reported 注册）：
- state_changed(new_state, old_state)
- count_changed(goods_id, new_count)
- result_reported(goods_id, ok, detail)
"""
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

from core.color_detector import ColorDetector
from core.goods_config import GoodsItem


class SorterState(Enum):
    """分拣状态机状态"""
    IDLE = auto()          # 空闲
    LOCKED = auto()        # 已锁定
    DISPATCHED = auto()    # 已下发
    WAIT_ACTION = auto()   # 等待动作完成
    WAIT_RESET = auto()    # 等待复位
    VERIFY = auto()        # 校验中
    SUCCESS = auto()       # 校验成功（瞬态，立即回空闲）
    FAIL = auto()          # 校验失败（瞬态，立即回空闲）


STATE_TEXT = {
    SorterState.IDLE: "空闲",
    SorterState.LOCKED: "已锁定",
    SorterState.DISPATCHED: "已下发",
    SorterState.WAIT_ACTION: "等待动作",
    SorterState.WAIT_RESET: "等待复位",
    SorterState.VERIFY: "校验中",
    SorterState.SUCCESS: "成功",
    SorterState.FAIL: "失败",
}


class Sorter:
    """分拣状态机：锁定目标 → 下发 → 等待动作/复位 → 视觉校验 → 计数"""

    def __init__(self, detector: ColorDetector, goods: List[GoodsItem],
                 tolerance_radius: float = 50.0,
                 frame_source: Optional[Callable[[], object]] = None):
        self.detector = detector
        self.goods = goods
        self.tolerance_radius = tolerance_radius          # 容差圈半径（像素）
        self.frame_source = frame_source                  # callable() -> 当前帧(BGR) 或 None
        self.state = SorterState.IDLE
        self.counts: Dict[str, int] = {g.goods_id: 0 for g in goods}
        self.target_goods_id: Optional[str] = None        # 当前锁定目标货物 ID
        self.target_color: Optional[str] = None           # 当前锁定目标颜色标识
        self.target_center: Optional[Tuple[float, float]] = None  # 中心像素坐标快照
        self.last_result: Optional[bool] = None           # 最近一次校验结果
        self._state_callbacks: List[Callable] = []
        self._count_callbacks: List[Callable] = []
        self._result_callbacks: List[Callable] = []

    # ---------- 回调注册 ----------
    def on_state_changed(self, cb: Callable) -> None:
        """注册状态变化回调 cb(new_state, old_state)"""
        self._state_callbacks.append(cb)

    def on_count_changed(self, cb: Callable) -> None:
        """注册计数变化回调 cb(goods_id, new_count)"""
        self._count_callbacks.append(cb)

    def on_result_reported(self, cb: Callable) -> None:
        """注册校验结果回调 cb(goods_id, ok, detail)"""
        self._result_callbacks.append(cb)

    # ---------- 对外事件 ----------
    def lock(self, goods_id: str, center: Tuple[float, float]) -> bool:
        """锁定目标货物：记录货物 ID + 中心像素坐标快照（仅空闲态可锁定）"""
        if self.state is not SorterState.IDLE:
            return False
        self.target_goods_id = goods_id
        self.target_color = self._color_of(goods_id)
        self.target_center = (float(center[0]), float(center[1]))
        self.last_result = None
        self._set_state(SorterState.LOCKED)
        return True

    def dispatch(self) -> bool:
        """下发分拣指令给机械臂（仅已锁定态可下发）"""
        if self.state is not SorterState.LOCKED:
            return False
        self._set_state(SorterState.DISPATCHED)
        return True

    def on_action_done(self) -> None:
        """机械臂动作完成事件（来自串口/模拟数据源）"""
        if self.state is not SorterState.DISPATCHED:
            return
        self._set_state(SorterState.WAIT_ACTION)

    def on_reset_done(self) -> None:
        """机械臂复位完成事件（来自串口/模拟数据源），随后自动进入校验"""
        if self.state is not SorterState.WAIT_ACTION:
            return
        self._set_state(SorterState.WAIT_RESET)
        self._verify()

    def verify(self, frame=None) -> None:
        """手动触发校验（可选；通常由复位完成事件自动触发）"""
        if self.state not in (SorterState.WAIT_RESET, SorterState.VERIFY):
            return
        self._verify(frame)

    def reset_counts(self) -> None:
        """清零所有货物计数"""
        for gid in self.counts:
            self.counts[gid] = 0
            self._emit_count(gid)

    def state_text(self) -> str:
        return STATE_TEXT.get(self.state, "未知")

    # ---------- 内部 ----------
    def _set_state(self, new_state: SorterState) -> None:
        old = self.state
        self.state = new_state
        for cb in self._state_callbacks:
            try:
                cb(new_state, old)
            except Exception:
                pass

    def _emit_count(self, goods_id: str) -> None:
        n = self.counts.get(goods_id, 0)
        for cb in self._count_callbacks:
            try:
                cb(goods_id, n)
            except Exception:
                pass

    def _emit_result(self, goods_id: str, ok: bool, detail: str) -> None:
        for cb in self._result_callbacks:
            try:
                cb(goods_id, ok, detail)
            except Exception:
                pass

    def _verify(self, frame=None) -> None:
        """校验：回看当前识别结果，检查锁定目标容差圈内是否仍有同色货物中心点"""
        self._set_state(SorterState.VERIFY)
        if frame is None and self.frame_source is not None:
            frame = self.frame_source()
        dets = self.detector.detect(frame) if frame is not None else []
        ok = True
        detail = "无同色货物残留在容差圈内"
        if self.target_center is not None and self.target_color is not None:
            tx, ty = self.target_center
            r = self.tolerance_radius
            for d in dets:
                if d.color_key == self.target_color:
                    cx, cy = d.center
                    if (cx - tx) ** 2 + (cy - ty) ** 2 <= r * r:
                        ok = False
                        detail = (f"容差圈({r:.0f}px)内仍检测到同色货物中心 "
                                  f"({cx:.0f},{cy:.0f})")
                        break
        self._finish(ok, detail)

    def _finish(self, ok: bool, detail: str) -> None:
        """结算：成功则计数 +1；随后清空锁定回空闲"""
        gid = self.target_goods_id
        self.last_result = ok
        if ok and gid is not None:
            self.counts[gid] += 1
            self._emit_count(gid)
        if gid is not None:
            self._emit_result(gid, ok, detail)
        self._set_state(SorterState.SUCCESS if ok else SorterState.FAIL)
        # 清空锁定，回空闲
        self.target_goods_id = None
        self.target_color = None
        self.target_center = None
        self._set_state(SorterState.IDLE)

    def _color_of(self, goods_id: str) -> Optional[str]:
        for g in self.goods:
            if g.goods_id == goods_id:
                return g.color_key
        return None
