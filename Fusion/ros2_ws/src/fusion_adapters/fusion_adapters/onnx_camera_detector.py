"""Pluggable ONNX camera cone detector (Ultralytics YOLO, NMS-free end2end export).

Mirrors the LiDAR adapter's philosophy: a ready-to-use default pipeline that publishes the
same `CameraConeDetections` contract as any third-party detector, so teams can swap it for
their own network by simply remapping topics (see `camera_detection_adapter.py`).

All geometry / class logic is ROS-independent and unit-testable (`OnnxConeDetector`).
Output classes are always mapped to the fusion convention [blue, yellow, orange, big_orange].
"""
from __future__ import annotations

import ast
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

try:
    from fusion_interfaces.msg import CameraConeDetection, CameraConeDetections
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image
except ImportError:  # pragma: no cover - bare-Python unit testing
    rclpy = None

    class Node:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    CameraConeDetection = CameraConeDetections = CompressedImage = Image = None  # type: ignore


# Fusion class order: 0 blue, 1 yellow, 2 small orange, 3 big orange.
DEFAULT_LABEL_TO_FUSION: Dict[str, int] = {
    "blue": 0, "blue_cone": 0,
    "yellow": 1, "yellow_cone": 1,
    "orange": 2, "orange_cone": 2, "small_orange_cone": 2,
    "big_orange": 3, "large_orange": 3, "large_orange_cone": 3, "big_orange_cone": 3,
}


def default_model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "models", "nano.onnx")


def _parse_names(raw: str) -> Dict[int, str]:
    try:
        parsed = ast.literal_eval(raw)
        return {int(k): str(v) for k, v in parsed.items()}
    except Exception:
        return {}


def _parse_imgsz(raw: str) -> Optional[Tuple[int, int]]:
    try:
        h, w = ast.literal_eval(raw)
        return int(h), int(w)
    except Exception:
        return None


class OnnxConeDetector:
    """Letterbox -> ONNX Runtime -> decode -> de-letterbox, no PyTorch/Ultralytics needed."""

    def __init__(
        self,
        model_path: str = "",
        conf_threshold: float = 0.25,
        providers: Optional[Sequence[str]] = None,
        label_map: Optional[Dict[str, int]] = None,
        intra_op_threads: int = 0,
        auto_rotate: bool = False,
    ) -> None:
        if ort is None or np is None or cv2 is None:
            raise RuntimeError("OnnxConeDetector requires numpy, opencv-python and onnxruntime")
        path = model_path or default_model_path()
        so = ort.SessionOptions()
        if intra_op_threads > 0:
            so.intra_op_num_threads = int(intra_op_threads)
        avail = ort.get_available_providers()
        chosen = [p for p in (providers or ("CUDAExecutionProvider", "CPUExecutionProvider")) if p in avail]
        self.session = ort.InferenceSession(path, sess_options=so, providers=chosen or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = int(inp.shape[2]), int(inp.shape[3])
        meta = self.session.get_modelmeta().custom_metadata_map
        names = _parse_names(meta.get("names", ""))
        lm = dict(DEFAULT_LABEL_TO_FUSION)
        if label_map:
            lm.update({k.lower(): v for k, v in label_map.items()})
        self.class_to_fusion: Dict[int, int] = {}
        for cid, name in names.items():
            if name.lower() in lm:
                self.class_to_fusion[cid] = lm[name.lower()]
        if not self.class_to_fusion:  # metadata missing -> assume Ultralytics FS order y,b,so,lo
            self.class_to_fusion = {0: 1, 1: 0, 2: 2, 3: 3}
        self.conf_threshold = float(conf_threshold)
        self.auto_rotate = bool(auto_rotate)
        self.end2end = meta.get("end2end", "False") == "True"
        self.last_inference_ms = 0.0

    @property
    def portrait_model(self) -> bool:
        return self.in_h > self.in_w

    def _letterbox(self, img, rotate: bool):
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
        scale = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.in_w - nw) // 2
        pad_y = (self.in_h - nh) // 2
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def detect(self, bgr) -> List[Tuple[float, float, float, float, List[float], float]]:
        """Return [(xmin, ymin, xmax, ymax, class_probs[4], confidence)] in ORIGINAL image pixels."""
        h0, w0 = bgr.shape[:2]
        # Optional: only needed if a network was trained on 90-degree rotated frames.
        # The bundled nano.onnx expects normal landscape frames letterboxed into its input.
        rotate = self.auto_rotate and self.portrait_model and w0 > h0
        canvas, scale, px, py = self._letterbox(bgr, rotate)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        self.last_inference_ms = (time.perf_counter() - t0) * 1e3
        return self._decode(out, scale, px, py, rotate, w0, h0)

    def _decode(self, out, scale, px, py, rotate, w0, h0):
        rows = out[0]
        results = []
        for x1, y1, x2, y2, conf, cls in rows:
            if conf < self.conf_threshold:
                continue
            fidx = self.class_to_fusion.get(int(cls))
            if fidx is None:
                continue
            bx1, by1 = (x1 - px) / scale, (y1 - py) / scale
            bx2, by2 = (x2 - px) / scale, (y2 - py) / scale
            if rotate:
                # Undo ROTATE_90_CLOCKWISE, which maps original (x, y) -> (h0 - y, x):
                #   original x = rotated y,  original y = h0 - rotated x
                x1o, x2o = by1, by2
                y1o, y2o = h0 - bx2, h0 - bx1
            else:
                x1o, y1o, x2o, y2o = bx1, by1, bx2, by2
            x1o, x2o = sorted((float(np.clip(x1o, 0, w0)), float(np.clip(x2o, 0, w0))))
            y1o, y2o = sorted((float(np.clip(y1o, 0, h0)), float(np.clip(y2o, 0, h0))))
            if x2o - x1o < 2.0 or y2o - y1o < 2.0:
                continue
            c = float(min(1.0, max(0.0, conf)))
            rest = (1.0 - c) / 3.0
            probs = [rest] * 4
            probs[fidx] = c
            results.append((x1o, y1o, x2o, y2o, probs, c))
        return results


class OnnxCameraDetectorNode(Node):
    """ROS 2 node: image topic -> CameraConeDetections (drop-in replacement for any detector)."""

    def __init__(self) -> None:
        super().__init__("onnx_camera_detector")
        self.declare_parameter("model_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("compressed", False)
        self.declare_parameter("output_topic", "/camera/cone_detections")
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("auto_rotate", False)
        self.declare_parameter("intra_op_threads", 0)
        self.declare_parameter("image_quality", 1.0)
        self.detector = OnnxConeDetector(
            model_path=str(self.get_parameter("model_path").value),
            conf_threshold=float(self.get_parameter("conf_threshold").value),
            auto_rotate=bool(self.get_parameter("auto_rotate").value),
            intra_op_threads=int(self.get_parameter("intra_op_threads").value),
        )
        self.image_quality = float(self.get_parameter("image_quality").value)
        topic = str(self.get_parameter("image_topic").value)
        self.pub = self.create_publisher(CameraConeDetections, str(self.get_parameter("output_topic").value), 10)
        if bool(self.get_parameter("compressed").value):
            self.create_subscription(CompressedImage, topic, self._on_compressed, 1)
        else:
            self.create_subscription(Image, topic, self._on_raw, 1)

    def _on_compressed(self, msg) -> None:
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            self._publish(msg.header, img)

    def _on_raw(self, msg) -> None:
        enc = msg.encoding.lower()
        ch = 4 if enc in ("bgra8", "rgba8") else 3
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, ch)
        if enc in ("rgb8", "rgba8"):
            img = img[:, :, 2::-1] if ch == 3 else img[:, :, [2, 1, 0]]
        else:
            img = img[:, :, :3]
        self._publish(msg.header, np.ascontiguousarray(img))

    def _publish(self, header, img) -> None:
        out = CameraConeDetections()
        out.header = header
        out.image_quality = float(self.image_quality)
        dets = []
        for x1, y1, x2, y2, probs, conf in self.detector.detect(img):
            d = CameraConeDetection()
            d.xmin, d.ymin, d.xmax, d.ymax = float(x1), float(y1), float(x2), float(y2)
            d.class_probs = [float(p) for p in probs]
            d.confidence = float(conf)
            dets.append(d)
        out.detections = dets
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OnnxCameraDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
