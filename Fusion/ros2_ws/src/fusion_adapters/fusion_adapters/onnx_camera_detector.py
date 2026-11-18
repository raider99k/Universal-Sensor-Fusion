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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def build_ort_provider_config(
    backend: str = "auto",
    available_providers: Optional[Sequence[str]] = None,
    enable_fp16: bool = True,
    enable_int8: bool = False,
    enable_dla: bool = False,
    dla_core: int = 0,
    engine_cache_dir: str = "/tmp/trt_cache",
) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
    """Build ONNX Runtime execution provider chain for CPU, CUDA, TensorRT, or Jetson DLA."""
    avail = list(available_providers) if available_providers is not None else (
        list(ort.get_available_providers()) if ort is not None else ["CPUExecutionProvider"]
    )
    backend_norm = str(backend or "auto").strip().lower()

    trt_options: Dict[str, Any] = {
        "trt_fp16_enable": bool(enable_fp16),
        "trt_int8_enable": bool(enable_int8),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(engine_cache_dir or "/tmp/trt_cache"),
    }
    if enable_dla:
        trt_options["trt_dla_enable"] = True
        trt_options["trt_dla_core"] = max(0, int(dla_core))

    if backend_norm == "cpu":
        ordered: List[Union[str, Tuple[str, Dict[str, Any]]]] = ["CPUExecutionProvider"]
    elif backend_norm == "cuda":
        ordered = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif backend_norm in ("tensorrt", "dla"):
        ordered = [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:  # "auto"
        ordered = [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    chosen: List[Union[str, Tuple[str, Dict[str, Any]]]] = []
    for entry in ordered:
        name = entry[0] if isinstance(entry, tuple) else entry
        if name in avail:
            chosen.append(entry)
    return chosen or ["CPUExecutionProvider"]


class OnnxConeDetector:
    """Letterbox -> ONNX Runtime -> decode -> de-letterbox, no PyTorch/Ultralytics needed."""

    def __init__(
        self,
        model_path: str = "",
        conf_threshold: float = 0.25,
        providers: Optional[Sequence[Union[str, Tuple[str, Dict[str, Any]]]]] = None,
        label_map: Optional[Dict[str, int]] = None,
        intra_op_threads: int = 0,
        auto_rotate: bool = False,
        backend: str = "auto",
        enable_fp16: bool = True,
        enable_int8: bool = False,
        enable_dla: bool = False,
        dla_core: int = 0,
        engine_cache_dir: str = "/tmp/trt_cache",
    ) -> None:
        if ort is None or np is None or cv2 is None:
            raise RuntimeError("OnnxConeDetector requires numpy, opencv-python and onnxruntime")
        path = model_path or default_model_path()
        so = ort.SessionOptions()
        if intra_op_threads > 0:
            so.intra_op_num_threads = int(intra_op_threads)
        avail = ort.get_available_providers()
        if providers is not None:
            chosen: List[Any] = [
                p for p in providers if (p[0] if isinstance(p, tuple) else p) in avail
            ] or ["CPUExecutionProvider"]
        else:
            chosen = build_ort_provider_config(
                backend=backend,
                available_providers=avail,
                enable_fp16=enable_fp16,
                enable_int8=enable_int8,
                enable_dla=enable_dla,
                dla_core=dla_core,
                engine_cache_dir=engine_cache_dir,
            )
        try:
            self.session = ort.InferenceSession(path, sess_options=so, providers=chosen)
        except Exception:
            self.session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.backend = str(backend or "auto")
        self.enable_dla = bool(enable_dla)
        self.dla_core = max(0, int(dla_core))
        self.active_providers: List[str] = list(self.session.get_providers())
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

    def detect_with_polygons(
        self, bgr, apex_width_ratio: float = 0.18
    ) -> List[Tuple[float, float, float, float, List[float], float, List[Tuple[float, float]]]]:
        """Return [(xmin, ymin, xmax, ymax, class_probs[4], confidence, polygon_uv)] in ORIGINAL pixels."""
        h0, w0 = bgr.shape[:2]
        rotate = self.auto_rotate and self.portrait_model and w0 > h0
        canvas, scale, px, py = self._letterbox(bgr, rotate)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        self.last_inference_ms = (time.perf_counter() - t0) * 1e3
        return self._decode_with_polygons(out, scale, px, py, rotate, w0, h0, apex_width_ratio)

    def _decode(self, out, scale, px, py, rotate, w0, h0):
        return [
            item[:6]
            for item in self._decode_with_polygons(out, scale, px, py, rotate, w0, h0, 0.18)
        ]

    def _decode_with_polygons(self, out, scale, px, py, rotate, w0, h0, apex_width_ratio: float):
        rows = out[0]
        results = []
        for row in rows:
            x1, y1, x2, y2, conf, cls = row[:6]
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

            poly_uv: List[Tuple[float, float]] = []
            extra = row[6:]
            if len(extra) >= 6:
                stride = 3 if len(extra) % 3 == 0 else (2 if len(extra) % 2 == 0 else 0)
                if stride > 0:
                    for k in range(0, len(extra), stride):
                        if stride == 3 and float(extra[k + 2]) < 0.25:
                            continue
                        kx = (float(extra[k]) - px) / scale
                        ky = (float(extra[k + 1]) - py) / scale
                        if rotate:
                            kx, ky = ky, h0 - kx
                        poly_uv.append((float(np.clip(kx, 0, w0)), float(np.clip(ky, 0, h0))))
            if len(poly_uv) < 3:
                poly_uv = cone_triangle_polygon_from_bbox(
                    x1o, y1o, x2o, y2o, apex_width_ratio=apex_width_ratio
                )

            c = float(min(1.0, max(0.0, conf)))
            rest = (1.0 - c) / 3.0
            probs = [rest] * 4
            probs[fidx] = c
            results.append((x1o, y1o, x2o, y2o, probs, c, poly_uv))
        return results


def cone_triangle_polygon_from_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    apex_width_ratio: float = 0.18,
) -> List[Tuple[float, float]]:
    """Construct a canonical 4-vertex truncated cone silhouette polygon from a 2D bbox."""
    cx = 0.5 * (float(xmin) + float(xmax))
    half_apex = 0.5 * max(0.0, float(xmax) - float(xmin)) * max(0.0, min(1.0, float(apex_width_ratio)))
    if half_apex <= 0.5:
        return [(cx, float(ymin)), (float(xmin), float(ymax)), (float(xmax), float(ymax))]
    return [
        (cx - half_apex, float(ymin)),
        (cx + half_apex, float(ymin)),
        (float(xmax), float(ymax)),
        (float(xmin), float(ymax)),
    ]


class TensorRTConeEngine(OnnxConeDetector):
    """Jetson Orin AGX TensorRT / DLA accelerated cone detector with safe CPU fallback."""

    def __init__(
        self,
        model_path: str = "",
        conf_threshold: float = 0.25,
        label_map: Optional[Dict[str, int]] = None,
        intra_op_threads: int = 0,
        auto_rotate: bool = False,
        enable_fp16: bool = True,
        enable_int8: bool = False,
        enable_dla: bool = False,
        dla_core: int = 0,
        engine_cache_dir: str = "/tmp/trt_cache",
    ) -> None:
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            label_map=label_map,
            intra_op_threads=intra_op_threads,
            auto_rotate=auto_rotate,
            backend="tensorrt",
            enable_fp16=enable_fp16,
            enable_int8=enable_int8,
            enable_dla=enable_dla,
            dla_core=dla_core,
            engine_cache_dir=engine_cache_dir,
        )

    @property
    def is_tensorrt_active(self) -> bool:
        return "TensorrtExecutionProvider" in self.active_providers

    @property
    def is_dla_enabled(self) -> bool:
        return self.enable_dla and self.is_tensorrt_active


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
        self.declare_parameter("backend", "auto")
        self.declare_parameter("enable_fp16", True)
        self.declare_parameter("enable_int8", False)
        self.declare_parameter("enable_dla", False)
        self.declare_parameter("dla_core", 0)
        self.declare_parameter("engine_cache_dir", "/tmp/trt_cache")
        self.detector = OnnxConeDetector(
            model_path=str(self.get_parameter("model_path").value),
            conf_threshold=float(self.get_parameter("conf_threshold").value),
            auto_rotate=bool(self.get_parameter("auto_rotate").value),
            intra_op_threads=int(self.get_parameter("intra_op_threads").value),
            backend=str(self.get_parameter("backend").value),
            enable_fp16=bool(self.get_parameter("enable_fp16").value),
            enable_int8=bool(self.get_parameter("enable_int8").value),
            enable_dla=bool(self.get_parameter("enable_dla").value),
            dla_core=int(self.get_parameter("dla_core").value),
            engine_cache_dir=str(self.get_parameter("engine_cache_dir").value),
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
