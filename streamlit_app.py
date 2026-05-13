from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st
import yaml


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from final_model_backend import run_backend_pipeline  # noqa: E402


MODEL_REF_CONFIG = REPO_ROOT / "configs" / "final_project_model.yaml"
STREAMLIT_OUTPUT_ROOT = REPO_ROOT / "data" / "inspection_outputs" / "streamlit_runs"
UPLOAD_ROOT = REPO_ROOT / "data" / "inspection_outputs" / "streamlit_uploads"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_ROOT) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def load_report(report_path: Path) -> dict:
    return json.loads(report_path.read_text(encoding="utf-8"))


def render_summary(summary: dict) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Detections", summary["detections"])
    metric_cols[1].metric("Tiles Used", summary["tile_count"])
    metric_cols[2].metric("Image Mode", "Tiled" if summary["used_tiling"] else "Single")
    metric_cols[3].metric("Model Run", summary["run_name"])


def render_detection_table(detections: list[dict]) -> None:
    rows = []
    for item in detections:
        rows.append(
            {
                "Class": item["class_name"],
                "Confidence": item["confidence"],
                "Severity Score": item["severity_score"],
                "Severity Band": item["severity_band"],
                "BBox XYXY": item["bbox_xyxy"],
                "Image Reference": item.get("image_reference", "uploaded image"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="PCB Defect Detector", layout="wide")

    model_ref = load_yaml(MODEL_REF_CONFIG)

    st.title("PCB Defect Detector")
    st.caption("Upload one PCB image, run the frozen final detector, and review defect severity in a technician-friendly report.")

    with st.sidebar:
        st.subheader("Current Backbone")
        st.markdown(f"- Run: `{model_ref['selected_run_name']}`")
        st.markdown(f"- Checkpoint: `{model_ref['selected_checkpoint']}`")
        st.markdown(f"- Target dataset: `{model_ref['target_dataset']}`")
        st.markdown("---")
        st.subheader("Instructions")
        st.markdown("1. Upload one PCB image.")
        st.markdown("2. Click **Run Detection**.")
        st.markdown("3. Review the annotated image, defect table, and downloadable reports.")
        st.markdown("---")
        st.subheader("Advanced Settings")
        conf = st.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        tile_size = st.selectbox("Tile Size", options=[640, 800, 1024], index=0)
        tile_overlap = st.selectbox("Tile Overlap", options=[64, 128, 192], index=1)
        merge_iou = st.slider("Merge IoU", min_value=0.10, max_value=0.70, value=0.30, step=0.05)

    uploaded_file = st.file_uploader(
        "Upload a PCB image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Use one PCB image. Large images are tiled automatically before inference.",
    )

    if uploaded_file is not None:
        left_col, right_col = st.columns([1, 1])
        with left_col:
            st.subheader("Uploaded Image")
            st.image(uploaded_file, use_container_width=True)
        with right_col:
            st.subheader("What You Will Get")
            st.markdown("- Annotated PCB image")
            st.markdown("- Defect table with confidence and severity")
            st.markdown("- Downloadable JSON report")
            st.markdown("- Downloadable CSV report")

    run_clicked = st.button("Run Detection", type="primary", disabled=uploaded_file is None)

    if run_clicked and uploaded_file is not None:
        saved_upload = save_uploaded_file(uploaded_file)
        with st.spinner("Running final PCB defect detector..."):
            result = run_backend_pipeline(
                image_path=saved_upload,
                model_ref_config=MODEL_REF_CONFIG,
                output_root=STREAMLIT_OUTPUT_ROOT,
                conf=conf,
                device="cpu",
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                merge_iou=merge_iou,
            )

        report = load_report(result["json_path"])
        summary = report["summary"]
        detections = report["detections"]

        st.success("Detection complete.")
        render_summary(summary)

        image_col, report_col = st.columns([1.15, 1.0])
        with image_col:
            st.subheader("Annotated Result")
            st.image(str(result["overlay_path"]), use_container_width=True)

        with report_col:
            st.subheader("Report Downloads")
            st.download_button(
                "Download JSON Report",
                data=result["json_path"].read_bytes(),
                file_name=result["json_path"].name,
                mime="application/json",
            )
            st.download_button(
                "Download CSV Report",
                data=result["csv_path"].read_bytes(),
                file_name=result["csv_path"].name,
                mime="text/csv",
            )
            st.markdown("---")
            st.markdown(f"**Output folder:** `{result['output_dir']}`")
            st.markdown(f"**Annotated image:** `{result['overlay_path'].name}`")

        st.subheader("Detected Defects")
        if detections:
            render_detection_table(detections)
        else:
            st.info("No defects were detected in this image at the current confidence threshold.")

        with st.expander("Backend Details"):
            st.json(
                {
                    "used_tiling": result["used_tiling"],
                    "tile_count": result["tile_count"],
                    "json_report": str(result["json_path"]),
                    "csv_report": str(result["csv_path"]),
                }
            )


if __name__ == "__main__":
    main()
