#!/usr/bin/env python3
"""Publish YOLO detections with OpenCV refined rotated boxes and grasp centers."""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "vision_sorter"))

import config as C  # noqa: E402
import grasp_refine  # noqa: E402
from yolov8_bpu import Yolov8Bpu  # noqa: E402


class YoloOnlyPreview(Node):
    def __init__(self, args):
        super().__init__("yolo_only_preview")
        self.args = args
        self.detector = Yolov8Bpu(
            args.bin,
            C.YOLO_CLASS_NAMES,
            conf_thres=args.conf,
            iou_thres=0.45,
            verbose=True,
        )
        self.pub = self.create_publisher(CompressedImage, args.output_topic, 10)
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(CompressedImage, args.input_topic, self.on_image, qos)
        self.last_run = 0.0
        self.count = 0
        self.last_log = time.time()
        self.last_count = 0
        self.get_logger().info(
            "YOLO-only preview: model=%s conf=%.2f input=%s output=%s"
            % (args.bin, args.conf, args.input_topic, args.output_topic)
        )

    def on_image(self, msg):
        now = time.time()
        if now - self.last_run < 1.0 / self.args.max_fps:
            return
        self.last_run = now
        buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            return
        detections = sorted(
            self.detector.detect(frame, conf_thres=self.args.conf),
            key=lambda item: -item.score,
        )
        canvas = frame.copy()
        refined_count = 0
        for item in detections:
            raw_bbox = [item.x1, item.y1, item.x2, item.y2]
            _shape, color = C.parse_yolo_class(item.name)
            gp = None
            if color is not None and getattr(C, "DEBUG_BOX_FIT", True):
                try:
                    gp = grasp_refine.refine(frame, raw_bbox, color)
                except Exception as exc:                    # noqa: BLE001
                    self.get_logger().warn(
                        "OpenCV refine failed for %s: %s: %s"
                        % (item.name, type(exc).__name__, exc),
                        throttle_duration_sec=5.0,
                    )

            p1 = (int(round(item.x1)), int(round(item.y1)))
            p2 = (int(round(item.x2)), int(round(item.y2)))
            fitted = bool(gp and gp.get("ok"))
            if fitted:
                refined_count += 1
                # Keep the raw YOLO box visible in yellow and draw the OpenCV
                # minimum-area rotated rectangle in green.
                cv2.rectangle(canvas, p1, p2, (0, 220, 255), 1)
                points = np.asarray(gp["rect_points"], np.float32).copy()
                points[:, 0] = np.clip(points[:, 0], 0, frame.shape[1] - 1)
                points[:, 1] = np.clip(points[:, 1], 0, frame.shape[0] - 1)
                cv2.polylines(
                    canvas, [points.astype(np.int32).reshape(-1, 1, 2)],
                    True, (0, 255, 0), 2, cv2.LINE_AA,
                )
                center = tuple(float(v) for v in gp["point"])
                center_color = (0, 0, 255)
                suffix = "R"
            else:
                cv2.rectangle(canvas, p1, p2, (0, 255, 0), 1)
                center = (
                    float(gp["point"][0]) if gp is not None
                    else (item.x1 + item.x2) / 2.0,
                    float(gp["point"][1]) if gp is not None
                    else (item.y1 + item.y2) / 2.0,
                )
                center_color = (0, 165, 255)
                suffix = "raw"

            cx, cy = int(round(center[0])), int(round(center[1]))
            cv2.drawMarker(
                canvas, (cx, cy), center_color,
                cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA,
            )
            cv2.circle(canvas, (cx, cy), 2, center_color, -1, cv2.LINE_AA)

            label = "%s %.2f %s" % (item.name, item.score, suffix)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1
            )
            top = max(0, p1[1] - th - 7)
            cv2.rectangle(
                canvas, (p1[0], top), (p1[0] + tw + 4, top + th + 6),
                (0, 255, 0), -1,
            )
            cv2.putText(
                canvas, label, (p1[0] + 2, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "YOLO+OpenCV  boxes=%d  rotate=%d  conf>=%.2f"
            % (len(detections), refined_count, self.args.conf),
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.quality]
        )
        if not ok:
            return
        output = CompressedImage()
        output.header = msg.header
        output.format = "jpeg"
        output.data = encoded.tobytes()
        self.pub.publish(output)

        self.count += 1
        if now - self.last_log >= 5.0:
            fps = (self.count - self.last_count) / (now - self.last_log)
            self.get_logger().info(
                "YOLO+OpenCV %.1f fps, boxes=%d, refined=%d, top=%s"
                % (
                    fps,
                    len(detections),
                    refined_count,
                    ", ".join(
                        "%s:%.3f" % (item.name, item.score)
                        for item in detections[:4]
                    ) or "none",
                )
            )
            self.last_log = now
            self.last_count = self.count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=C.YOLO_BIN_MODEL)
    parser.add_argument("--input-topic", default="/image")
    parser.add_argument("--output-topic", default="/image_yolo_raw")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()

    rclpy.init()
    node = YoloOnlyPreview(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
