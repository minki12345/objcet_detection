import io
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@st.cache_resource
def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Load YOLOv8 model once and reuse across reruns."""
    model = YOLO(model_path)
    return model


def preprocess_image(image: Image.Image, max_size: int = 1280) -> np.ndarray:
    """Convert PIL image to resized BGR numpy array for OpenCV/YOLO."""
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    scale = min(max_size / max(width, 1), max_size / max(height, 1), 1.0)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = rgb_image.resize((new_width, new_height))
    np_image = np.array(resized)
    bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def run_detection(
    model: YOLO, bgr_image: np.ndarray, conf_threshold: float = 0.25
) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, int]]:
    """Run YOLO detection on CPU and return detections + counts."""
    results = model.predict(
        source=bgr_image,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )
    result = results[0]

    detections: List[Dict[str, float]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        label = result.names[class_id]
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    object_counts = dict(Counter(item["label"] for item in detections))
    return bgr_image, detections, object_counts


def draw_boxes(bgr_image: np.ndarray, detections: List[Dict[str, float]]) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    drawn = bgr_image.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        label = str(det["label"])
        confidence = float(det["confidence"])
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            drawn,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return drawn


def display_results(
    result_bgr: np.ndarray, detections: List[Dict[str, float]], object_counts: Dict[str, int]
) -> None:
    """Render result image and object details in Streamlit."""
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    st.subheader("Detection Result")
    st.image(result_rgb, caption="Detected Objects", use_container_width=True)

    st.subheader("Detected Object List")
    if not detections:
        st.info("No objects detected.")
        return

    for idx, det in enumerate(detections, start=1):
        st.write(
            f"{idx}. **{det['label']}** - Confidence: {det['confidence']:.2f} "
            f"(bbox: {det['x1']}, {det['y1']}, {det['x2']}, {det['y2']})"
        )

    st.subheader("Object Count Statistics")
    st.json(object_counts)


def _bytes_to_pil(uploaded_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")


def main() -> None:
    st.set_page_config(page_title="Mobile Object Detection App", page_icon="📱", layout="centered")
    st.title("📱 Mobile Object Detection App")
    st.write(
        "Capture an image with your smartphone camera or upload one, then run YOLOv8 "
        "to detect objects with bounding boxes, labels, confidence, and counts."
    )

    st.subheader("Input Section")
    camera_image = st.camera_input("Take a photo")
    uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png", "webp"])
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

    selected_pil = None
    if camera_image is not None:
        selected_pil = _bytes_to_pil(camera_image.getvalue())
        st.caption("Using camera image.")
    elif uploaded_file is not None:
        selected_pil = _bytes_to_pil(uploaded_file.read())
        st.caption("Using uploaded image.")

    if selected_pil is not None:
        st.image(selected_pil, caption="Input Image", use_container_width=True)

    if st.button("Run Detection", type="primary"):
        if selected_pil is None:
            st.warning("Please capture or upload an image first.")
            return

        with st.spinner("Loading model and running inference on CPU..."):
            model = load_model("yolov8n.pt")
            preprocessed = preprocess_image(selected_pil, max_size=1280)
            base_bgr, detections, object_counts = run_detection(model, preprocessed, conf_threshold)
            output_bgr = draw_boxes(base_bgr, detections)

        display_results(output_bgr, detections, object_counts)

        st.markdown("---")
        st.subheader("Possible Extensions")
        st.markdown(
            "- AI explanation of detected scenes\n"
            "- Dangerous object alerting\n"
            "- Real-time video detection\n"
            "- Jetson deployment optimization"
        )


if __name__ == "__main__":
    main()
