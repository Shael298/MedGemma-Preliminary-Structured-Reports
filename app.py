import os

import streamlit as st
from dotenv import load_dotenv

from medgemma_service import (
    draft_report,
    load_image_from_bytes,
    load_medgemma,
)

load_dotenv()

DEFAULT_MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(
    page_title="MedGemma Preliminary Reporter",
    page_icon="M",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading MedGemma model...")
def get_model_bundle(model_id: str, hf_token: str | None):
    return load_medgemma(model_id=model_id, hf_token=hf_token)


st.title("MedGemma Medical Scan Reviewer")
st.caption(
    "Upload a chest X-ray and generate a preliminary structured draft for clinician review."
)
st.warning(
    "This app creates AI-generated preliminary drafts only. It is not a final diagnostic tool."
)
st.info(
    "Current release scope is chest X-ray only. Upload a single chest radiograph image "
    "such as PNG, JPG, or JPEG. These are the image formats exercised in the current test pass."
)

with st.sidebar:
    st.header("Model Settings")
    model_id = st.text_input("Model ID", value=DEFAULT_MODEL_ID)

    st.header("Case Details")
    # Active project scope is intentionally narrow while the report pipeline is stabilized.
    study_type = "Chest X-ray"
    st.text_input("Study Type", value=study_type, disabled=True)
    clinical_context = st.text_area(
        "Clinical Context",
        placeholder="Example: Cough, fever, evaluate for focal consolidation.",
    )
    comparison = st.text_input(
        "Comparison",
        value="No comparison study provided.",
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    try:
        image = load_image_from_bytes(file_bytes, uploaded_file.name)
    except Exception as exc:
        st.error(f"Could not read uploaded image: {exc}")
        st.stop()

    preview_column, report_column = st.columns([1, 1.4])

    with preview_column:
        st.subheader("Image Preview")
        st.image(image, width="stretch")
        st.caption(f"File: {uploaded_file.name}")

    with report_column:
        st.subheader("Structured Draft")
        generate_clicked = st.button("Generate Preliminary Report", type="primary")

        if generate_clicked:
            with st.spinner("Running MedGemma inference..."):
                try:
                    bundle = get_model_bundle(model_id=model_id, hf_token=HF_TOKEN)

                    report = draft_report(
                        bundle=bundle,
                        image=image,
                        study_type=study_type,
                        clinical_context=clinical_context,
                        comparison=comparison,
                    )
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
                else:
                    if bundle.quantized_4bit:
                        st.caption(
                            f"Loaded in 4-bit mode on {bundle.device} with {bundle.compute_dtype} compute."
                        )
                    else:
                        st.caption(
                            f"Loaded without 4-bit quantization on {bundle.device} using {bundle.compute_dtype}."
                        )
                    st.markdown(report.to_markdown())
else:
    st.caption("Upload a chest X-ray image to begin.")
