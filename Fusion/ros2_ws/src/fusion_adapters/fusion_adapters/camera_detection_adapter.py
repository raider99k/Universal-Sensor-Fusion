from __future__ import annotations

from typing import List, Optional, Tuple

try:
    from fusion_interfaces.msg import CameraConeDetection, CameraConeDetections
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32
    from vision_msgs.msg import Detection2DArray
except ImportError:  # pragma: no cover - fallback for bare-Python unit testing
    rclpy = None

    class Node:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class _Header:
        def __init__(self):
            self.frame_id = ""

    class CameraConeDetection:  # type: ignore[no-redef]
        def __init__(self):
            self.xmin = 0.0
            self.ymin = 0.0
            self.xmax = 0.0
            self.ymax = 0.0
            self.class_probs = [0.0, 0.0, 0.0, 1.0]
            self.confidence = 0.0

    class CameraConeDetections:  # type: ignore[no-redef]
        def __init__(self):
            self.header = _Header()
            self.image_quality = 1.0
            self.detections = []

    class Float32:  # type: ignore[no-redef]
        def __init__(self, data: float = 0.0):
            self.data = data

    class Detection2DArray:  # type: ignore[no-redef]
        def __init__(self):
            self.header = _Header()
            self.detections = []


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_getattr(obj, name: str, default):
    return getattr(obj, name, default) if obj is not None else default


def _extract_bbox(det) -> Tuple[float, float, float, float]:
    bbox = _safe_getattr(det, "bbox", None)
    center = _safe_getattr(bbox, "center", None)
    size_x = float(_safe_getattr(bbox, "size_x", 0.0))
    size_y = float(_safe_getattr(bbox, "size_y", 0.0))

    # vision_msgs variants use either center.x/y or center.position.x/y.
    cx = _safe_getattr(center, "x", None)
    cy = _safe_getattr(center, "y", None)
    if cx is None or cy is None:
        position = _safe_getattr(center, "position", None)
        cx = _safe_getattr(position, "x", 0.0)
        cy = _safe_getattr(position, "y", 0.0)

    cx = float(cx if cx is not None else 0.0)
    cy = float(cy if cy is not None else 0.0)
    xmin = cx - 0.5 * size_x
    ymin = cy - 0.5 * size_y
    xmax = cx + 0.5 * size_x
    ymax = cy + 0.5 * size_y
    return xmin, ymin, xmax, ymax


class ClassMapper:
    def __init__(self, mapping_expr: str) -> None:
        self.label_to_idx = {}
        self._parse_mapping(mapping_expr)

    def _parse_mapping(self, mapping_expr: str) -> None:
        self.label_to_idx.clear()
        chunks = [c.strip() for c in mapping_expr.split(",") if c.strip()]
        for chunk in chunks:
            if ":" not in chunk:
                continue
            label, idx = chunk.split(":", 1)
            label = label.strip().lower()
            try:
                idx_i = int(idx.strip())
            except ValueError:
                continue
            if 0 <= idx_i <= 3:
                self.label_to_idx[label] = idx_i

    def to_index(self, class_id: str) -> Optional[int]:
        label = str(class_id).strip().lower()
        if label in self.label_to_idx:
            return self.label_to_idx[label]

        # Allow numeric class identifiers in upstream detections.
        try:
            numeric = int(label)
            if 0 <= numeric <= 3:
                return numeric
        except ValueError:
            pass
        return None


def _extract_label_score(result) -> Tuple[str, float]:
    # vision_msgs/ObjectHypothesisWithPose: result.hypothesis.class_id / score
    hypothesis = _safe_getattr(result, "hypothesis", None)
    if hypothesis is not None:
        class_id = str(_safe_getattr(hypothesis, "class_id", "unknown"))
        score = float(_safe_getattr(hypothesis, "score", 0.0))
        return class_id, _clamp(score, 0.0, 1.0)

    # Fallback for older variants.
    class_id = str(_safe_getattr(result, "id", "unknown"))
    score = float(_safe_getattr(result, "score", 0.0))
    return class_id, _clamp(score, 0.0, 1.0)


class CameraDetectionAdapter(Node):
    def __init__(self) -> None:
        super().__init__("camera_detection_adapter")

        self.declare_parameter("input_topic", "/vision/cone_detections")
        self.declare_parameter("output_topic", "/camera/cone_detections")
        self.declare_parameter("quality_topic", "")
        self.declare_parameter("default_image_quality", 1.0)
        self.declare_parameter(
            "class_map",
            "blue:0,blue_cone:0,yellow:1,yellow_cone:1,orange:2,orange_cone:2,big_orange:2,large_orange:2,large_orange_cone:2,red:2,unknown:3",
        )
        self.declare_parameter("max_detections", 256)
        self.declare_parameter("min_confidence", 0.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.quality_topic = str(self.get_parameter("quality_topic").value)
        self.default_image_quality = float(
            self.get_parameter("default_image_quality").value
        )
        self.class_mapper = ClassMapper(str(self.get_parameter("class_map").value))
        self.max_detections = int(self.get_parameter("max_detections").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)

        self.last_image_quality = _clamp(self.default_image_quality, 0.0, 1.0)
        self.pub = self.create_publisher(CameraConeDetections, self.output_topic, 10)
        self.sub = self.create_subscription(
            Detection2DArray, self.input_topic, self._on_detections, 10
        )
        self.quality_sub = None
        if self.quality_topic:
            self.quality_sub = self.create_subscription(
                Float32, self.quality_topic, self._on_quality, 10
            )

        self.get_logger().info(
            f"camera_detection_adapter: {self.input_topic} -> {self.output_topic}"
        )
        self.get_logger().warn(
            "Implementation status: prototype adapter. Validate class-map and detection schema "
            "against your deployed camera detector before production use."
        )

    def _on_quality(self, msg: Float32) -> None:
        self.last_image_quality = _clamp(float(msg.data), 0.0, 1.0)

    def _on_detections(self, msg: Detection2DArray) -> None:
        out = CameraConeDetections()
        out.header = msg.header
        out.image_quality = float(self.last_image_quality)

        detections_out: List[CameraConeDetection] = []
        for det in msg.detections[: self.max_detections]:
            probs = [0.0, 0.0, 0.0, 0.0]
            best_conf = 0.0

            for result in det.results:
                label, score = _extract_label_score(result)
                idx = self.class_mapper.to_index(label)
                if idx is None:
                    continue
                probs[idx] += score
                best_conf = max(best_conf, score)

            total = sum(probs)
            if total <= 1e-9:
                probs = [0.0, 0.0, 0.0, 1.0]
            else:
                probs = [v / total for v in probs]

            confidence = float(_clamp(best_conf, 0.0, 1.0))
            if confidence < self.min_confidence:
                continue

            xmin, ymin, xmax, ymax = _extract_bbox(det)

            out_det = CameraConeDetection()
            out_det.xmin = float(xmin)
            out_det.ymin = float(ymin)
            out_det.xmax = float(xmax)
            out_det.ymax = float(ymax)
            out_det.class_probs = [float(v) for v in probs]
            out_det.confidence = confidence
            detections_out.append(out_det)

        out.detections = detections_out
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraDetectionAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
