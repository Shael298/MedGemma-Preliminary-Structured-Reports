import io
import json
import re
import sys
from dataclasses import dataclass

import numpy as np
import pydicom
import torch
from json_repair import repair_json
from PIL import Image, ImageFile, ImageOps
from pydantic import ValidationError
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from schemas import StructuredReport

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPORT_TEMPLATE = """
STUDY_TYPE: <one line>
CLINICAL_CONTEXT: <one line>
TECHNIQUE: <one line>
COMPARISON: <one line>
FINDINGS:
- <bullet>
IMPRESSION:
- <bullet>
URGENCY: routine|expedited|critical|unknown
CONFIDENCE: <number from 0 to 1>
LIMITATIONS:
- <bullet>
FOLLOW_UP:
- <bullet>
""".strip()

FOLLOW_UP_GUIDANCE = (
    "For chest radiographs suspicious for infective pneumonia, consider a follow-up chest "
    "radiograph in about 6 weeks when clinically appropriate, especially if symptoms persist "
    "or the patient has higher risk for underlying malignancy. If the opacity does not resolve, "
    "further evaluation such as CT chest may be needed to exclude an underlying lesion."
)


@dataclass(frozen=True)
class LoadedMedGemma:
    model: AutoModelForImageTextToText
    processor: AutoProcessor
    device: str
    quantized_4bit: bool
    compute_dtype: str


class MedGemmaLoadError(RuntimeError):
    pass


def normalize_to_uint8(pixel_array: np.ndarray, invert: bool = False) -> np.ndarray:
    array = pixel_array.astype(np.float32)
    array -= array.min()
    peak = array.max()
    if peak > 0:
        array /= peak
    if invert:
        array = 1.0 - array
    return (array * 255).clip(0, 255).astype(np.uint8)


def coerce_pixel_array_to_image(pixel_array: np.ndarray, invert: bool = False) -> Image.Image:
    array = np.asarray(pixel_array)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] not in (3, 4) and array.shape[0] not in (3, 4):
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)

    if array.ndim == 2:
        normalized = normalize_to_uint8(array, invert=invert)
        return Image.fromarray(normalized).convert("RGB")

    if array.ndim == 3 and array.shape[-1] in (3, 4):
        normalized = normalize_to_uint8(array, invert=False)
        mode = "RGBA" if normalized.shape[-1] == 4 else "RGB"
        return Image.fromarray(normalized, mode=mode).convert("RGB")

    raise ValueError(f"Unsupported image shape for conversion: {array.shape}")


def load_image_from_bytes(file_bytes: bytes, file_name: str) -> Image.Image:
    suffix = file_name.lower().rsplit(".", maxsplit=1)[-1]
    if suffix == "dcm":
        dataset = pydicom.dcmread(io.BytesIO(file_bytes))
        pixel_array = dataset.pixel_array
        invert = getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1"
        return coerce_pixel_array_to_image(pixel_array, invert=invert)

    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def get_runtime_diagnostics() -> dict[str, str]:
    diagnostics = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
    }

    if torch.cuda.is_available():
        diagnostics["gpu_name"] = torch.cuda.get_device_name(0)
        diagnostics["bf16_supported"] = str(torch.cuda.is_bf16_supported())
        diagnostics["gpu_total_memory_gb"] = (
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f}"
        )
    else:
        diagnostics["gpu_name"] = "None"
        diagnostics["bf16_supported"] = "False"
        diagnostics["gpu_total_memory_gb"] = "0.00"

    try:
        import bitsandbytes as bnb
    except Exception as exc:
        diagnostics["bitsandbytes_available"] = "False"
        diagnostics["bitsandbytes_version"] = f"unavailable: {exc}"
    else:
        diagnostics["bitsandbytes_available"] = "True"
        diagnostics["bitsandbytes_version"] = bnb.__version__

    return diagnostics


def format_diagnostics(diagnostics: dict[str, str]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in diagnostics.items())


def load_medgemma(
    model_id: str,
    hf_token: str | None = None,
) -> LoadedMedGemma:
    diagnostics = get_runtime_diagnostics()
    use_cuda = torch.cuda.is_available()
    can_quantize_4bit = use_cuda

    if use_cuda and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif use_cuda:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    quantization_config = None
    if can_quantize_4bit:
        if diagnostics["bitsandbytes_available"] != "True":
            raise MedGemmaLoadError(
                "4-bit loading requires bitsandbytes, but it is not available.\n"
                f"{format_diagnostics(diagnostics)}"
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )

    model_load_kwargs = {
        "token": hf_token,
        "device_map": "auto" if use_cuda else None,
        "low_cpu_mem_usage": True,
    }
    quantized_4bit = quantization_config is not None
    if quantized_4bit:
        model_load_kwargs["quantization_config"] = quantization_config
    else:
        model_load_kwargs["torch_dtype"] = model_dtype

    try:
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            **model_load_kwargs,
        )
    except Exception as exc:
        raise MedGemmaLoadError(
            "Failed to load MedGemma. Common causes are missing Hugging Face access, "
            "unsupported bitsandbytes/CUDA setup, or running out of memory during load.\n"
            f"model_id: {model_id}\n"
            f"target_quantized_4bit: {can_quantize_4bit}\n"
            f"compute_dtype: {str(model_dtype).replace('torch.', '')}\n"
            f"{format_diagnostics(diagnostics)}"
        ) from exc

    device = "cuda" if use_cuda else "cpu"
    if not use_cuda:
        model.to(device)

    model.eval()
    return LoadedMedGemma(
        model=model,
        processor=processor,
        device=device,
        quantized_4bit=quantized_4bit,
        compute_dtype=str(model_dtype).replace("torch.", ""),
    )


def build_prompt(
    study_type: str,
    clinical_context: str,
    comparison: str,
) -> str:
    return f"""
You are assisting with preliminary medical image review.
Draft a structured report from the uploaded image for clinician review only.
Do not present the output as a final diagnosis.
If the image is insufficient, say so in the limitations field.
Do not include chain-of-thought, hidden reasoning, planning text, markdown fences, or commentary.
Return exactly one plain-text report using this template and nothing before or after it:
{REPORT_TEMPLATE}
Every section header must start on its own new line.
Do not place two section headers on the same line.
Use exactly these header names: STUDY_TYPE, CLINICAL_CONTEXT, TECHNIQUE, COMPARISON, FINDINGS, IMPRESSION, URGENCY, CONFIDENCE, LIMITATIONS, FOLLOW_UP.
Write CONFIDENCE as a decimal number between 0 and 1, for example 0.7.
For FINDINGS, IMPRESSION, LIMITATIONS, and FOLLOW_UP, put each item on its own bullet line starting with "- ".

Use these inputs:
- study_type: {study_type or "Unknown study"}
- clinical_context: {clinical_context or "Not provided"}
- comparison: {comparison or "No comparison study provided"}
- technique: infer only if visible; otherwise say "Not clearly identifiable from the single uploaded image."
Follow-up guidance: {FOLLOW_UP_GUIDANCE}
""".strip()


def build_case_context(
    study_type: str,
    clinical_context: str,
    comparison: str,
    findings: list[str] | None = None,
    impression: list[str] | None = None,
) -> str:
    parts = [
        "You are assisting with preliminary chest X-ray review.",
        f"study_type: {study_type or 'Unknown study'}",
        f"clinical_context: {clinical_context or 'Not provided'}",
        f"comparison: {comparison or 'No comparison study provided'}",
    ]
    if findings:
        parts.append("current_findings:")
        parts.extend(f"- {item}" for item in findings)
    if impression:
        parts.append("current_impression:")
        parts.extend(f"- {item}" for item in impression)
    return "\n".join(parts)


def build_section_prompt(
    section_name: str,
    study_type: str,
    clinical_context: str,
    comparison: str,
    findings: list[str] | None = None,
    impression: list[str] | None = None,
) -> str:
    context = build_case_context(study_type, clinical_context, comparison, findings, impression)
    prompts = {
        "technique": (
            "Return one short line only for the imaging technique.\n"
            'If unclear, return exactly: Not clearly identifiable from the single uploaded image.'
        ),
        "findings": (
            "Return only FINDINGS bullet points.\n"
            "Rules:\n"
            '- Return 2 to 6 bullet lines only, each starting with "- ".\n'
            "- No heading.\n"
            "- No plan, no reasoning, no explanation.\n"
            "- Mention only visible imaging findings."
        ),
        "impression": (
            "Return only IMPRESSION bullet points.\n"
            "Rules:\n"
            '- Return 1 to 2 bullet lines only, each starting with "- ".\n'
            "- No heading.\n"
            "- No plan, no reasoning, no explanation.\n"
            "- Summarize the most important abnormal conclusion only.\n"
            '- Do not restate normal findings such as clear lungs or normal heart size.\n'
            '- If there is no convincing acute abnormality, return exactly: "- No acute cardiopulmonary abnormality identified."'
        ),
        "urgency": (
            "Return exactly one word only from this list: routine, expedited, critical, unknown.\n"
            "No punctuation. No explanation."
        ),
        "confidence": (
            "Return exactly one decimal number between 0 and 1 only.\n"
            "Example: 0.7\n"
            "Base the number on the current findings and impression.\n"
            "Do not return 0.0 unless the image is essentially unreadable.\n"
            "No words. No explanation."
        ),
        "limitations": (
            "Return only LIMITATIONS bullet points.\n"
            "Rules:\n"
            '- Return 0 to 2 bullet lines only, each starting with "- ".\n'
            "- No heading.\n"
            "- No plan, no reasoning, no explanation.\n"
            "- Mention actual limitations such as single view, incomplete visualization, or uncertainty.\n"
            '- If there are no meaningful limitations, return exactly: "- None provided".'
        ),
        "follow_up": (
            "Return only FOLLOW_UP bullet points.\n"
            "Rules:\n"
            '- Return 0 to 2 bullet lines only, each starting with "- ".\n'
            "- No heading.\n"
            "- No plan, no reasoning, no explanation.\n"
            "- Every bullet must be an action or recommendation.\n"
            "- Do not restate findings or normal observations.\n"
            "- Keep recommendations short and clinically cautious.\n"
            "- Include a concrete timeframe when appropriate.\n"
            "- Prefer specific follow-up such as repeat chest radiograph in 6 weeks or CT chest if an opacity persists.\n"
            '- If no follow-up is needed, return exactly: "- None provided".'
        ),
    }
    instruction = prompts[section_name]
    return f"{context}\n\n{instruction}".strip()


def run_generation(
    bundle: LoadedMedGemma,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = bundle.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(bundle.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=bundle.model.dtype)

    prompt_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=5,
        )

    generated_tokens = generated[0][prompt_length:]
    return bundle.processor.decode(generated_tokens, skip_special_tokens=True).strip()


def strip_reasoning_preamble(text: str) -> str:
    cleaned = re.sub(r"<unused\d+>", "", text)
    cleaned = re.sub(r"```(?:text|json|python)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    lines = cleaned.splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        upper = line.upper()
        if not line:
            continue
        if line.startswith("-") or line.startswith("*"):
            return "\n".join(lines[index:]).strip()
        if upper.startswith(("TECHNIQUE", "FINDINGS", "IMPRESSION", "URGENCY", "CONFIDENCE", "LIMITATIONS", "FOLLOW_UP", "FOLLOW-UP")):
            return "\n".join(lines[index:]).strip()
        if not upper.startswith(("THOUGHT", "PLAN", "REFINEMENT", "**PLAN", "**REFINEMENT")):
            return "\n".join(lines[index:]).strip()
    return cleaned.strip()


def extract_bullet_list(text: str) -> list[str]:
    cleaned = strip_reasoning_preamble(text)
    items: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^[A-Z][A-Z_ \-]+:\s*$", line):
            continue
        if line.startswith(("- ", "* ")):
            candidate = line[2:].strip()
            if candidate and not re.fullmatch(r"[A-Z_]+", candidate):
                items.append(candidate)
    if items:
        return items

    first_line = cleaned.splitlines()[0].strip() if cleaned.splitlines() else cleaned.strip()
    first_line = re.sub(r"^[A-Z][A-Z_ \-]+:\s*", "", first_line).strip()
    if not first_line:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", first_line) if part.strip()]
    return [part for part in parts if not re.fullmatch(r"[A-Z_]+", part)]


def extract_scalar_text(text: str) -> str:
    cleaned = strip_reasoning_preamble(text)
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[A-Z][A-Z_ \-]+:\s*", "", line).strip()
        if line and not re.fullmatch(r"[A-Z_]+", line):
            return line
    return ""


def extract_urgency_value(text: str) -> str:
    cleaned = extract_scalar_text(text).lower()
    match = re.search(r"\b(routine|expedited|critical|unknown)\b", cleaned)
    return match.group(1) if match else "unknown"


def extract_confidence_value(text: str) -> float:
    cleaned = extract_scalar_text(text).lower()
    match = re.search(r"\b\d+(?:\.\d+)?%?\b", cleaned)
    if not match:
        return 0.0
    raw = match.group(0)
    if raw.endswith("%"):
        return max(0.0, min(1.0, float(raw[:-1]) / 100.0))
    value = float(raw)
    if value.is_integer() and 1.0 <= value <= 9.0:
        value = value / 10.0
    elif 1.0 < value <= 100.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def finalize_technique(study_type: str, technique_text: str) -> str:
    cleaned = extract_scalar_text(technique_text)
    if not cleaned:
        return "Not clearly identifiable from the single uploaded image."
    normalized = cleaned.strip().lower().rstrip(".")
    generic_values = {
        "x-ray",
        "chest x-ray",
        "radiograph",
        "chest radiograph",
        study_type.strip().lower().rstrip("."),
    }
    if normalized in generic_values:
        return "Not clearly identifiable from the single uploaded image."
    return cleaned


def clean_list_items(field_name: str, items: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    limits = {
        "findings": 6,
        "impression": 2,
        "limitations": 4,
        "follow_up": 2,
    }
    for item in items:
        candidate = item.strip().strip("-* ").strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        if re.fullmatch(r"[A-Z_]+", candidate):
            continue
        if lowered in {"none provided", "none"}:
            continue
        if field_name == "follow_up":
            words = re.findall(r"[a-zA-Z0-9]+", candidate)
            if len(words) < 3:
                continue
            if not re.search(
                r"\b(consider|recommend|follow[- ]?up|repeat|reassess|evaluate|evaluation|ct|radiograph|x-ray|imaging|correlat(?:e|ion)|monitor)\b",
                lowered,
            ):
                continue
            if lowered in {
                "consider",
                "recommend",
                "follow up",
                "follow-up",
                "correlate clinically",
            }:
                continue
            if not re.search(r"[.!?]$", candidate) and len(words) < 6:
                continue
        if field_name == "limitations":
            if "image quality" in lowered and "adequate" in lowered:
                continue
            if lowered.endswith("provided"):
                continue
        if field_name == "impression":
            if (
                "normal heart size" in lowered
                or "right lung appears clear" in lowered
                or "right lung clear" in lowered
                or "normal mediastinal" in lowered
                or "normal heart" in lowered
                or "clear" in lowered and "opacity" not in lowered and "consolid" not in lowered
            ):
                continue
        seen.add(lowered)
        cleaned.append(candidate)
        if len(cleaned) >= limits[field_name]:
            break
    return cleaned


def extract_candidate_json_text(text: str) -> str:
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return match.group(0)


def strip_json_line_comments(text: str) -> str:
    stripped_lines: list[str] = []
    for line in text.splitlines():
        if "//" not in line:
            stripped_lines.append(line)
            continue

        before, _, after = line.partition("//")
        # If the comment marker is inside a JSON string, keep the whole line.
        quote_count = len(re.findall(r'(?<!\\\\)"', before))
        if quote_count % 2 == 1:
            stripped_lines.append(line)
            continue

        stripped_lines.append(before.rstrip())
    return "\n".join(stripped_lines)


def normalize_key(key: str) -> str:
    normalized = key.strip().strip("'").strip('"')
    normalized = normalized.rstrip("\\")
    normalized = normalized.replace("\\_", "_")
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized.lower()


def coerce_to_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        trimmed = value.strip()
        return [trimmed] if trimmed else []
    trimmed = str(value).strip()
    return [trimmed] if trimmed else []


def normalize_report_payload(payload: dict) -> dict:
    aliases = {
        "clinical_image": "study_type",
        "clinicalimage": "study_type",
        "clinical_image_type": "study_type",
        "study": "study_type",
        "studytype": "study_type",
        "clinical_context": "clinical_context",
        "clinicalcontext": "clinical_context",
        "clinical_left": "clinical_context",
        "clinical_summary": "clinical_context",
        "context": "clinical_context",
        "indication": "clinical_context",
        "technicque": "technique",
        "technique": "technique",
        "tech": "technique",
        # Models often misuse "technical_details" to restate the modality (e.g. "Chest X-ray").
        # Treat it as study_type so we don't end up with technique="Chest X-ray".
        "technical_details": "study_type",
        "technicaldetail": "study_type",
        "comparision": "comparison",
        "comparison": "comparison",
        "finding": "findings",
        "findings": "findings",
        "impressions": "impression",
        "impression": "impression",
        "priority": "urgency",
        "urgency": "urgency",
        "confidence": "confidence",
        "confidence_score": "confidence",
        "confidencescore": "confidence",
        "confidence_level": "confidence",
        "limitation": "limitations",
        "limitations": "limitations",
        "limitatiions": "limitations",
        "image_quality": "limitations",
        "followup": "follow_up",
        "follow_up": "follow_up",
        "follow_up_guidance": "follow_up",
        "followup_guidance": "follow_up",
    }
    known_fields = {
        "study_type",
        "clinical_context",
        "technique",
        "comparison",
        "findings",
        "impression",
        "urgency",
        "confidence",
        "limitations",
        "follow_up",
    }

    def is_empty(value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, dict):
            return len(value) == 0
        return False

    def parse_confidence(value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            num = float(value)
        else:
            raw = str(value).strip().lower()
            if not raw:
                return 0.0
            if raw in {"low", "low confidence"}:
                return 0.3
            if raw in {"medium", "moderate", "mid", "medium confidence"}:
                return 0.6
            if raw in {"high", "high confidence"}:
                return 0.85
            raw = raw.replace("percent", "%")
            if raw.endswith("%"):
                try:
                    num = float(raw[:-1].strip()) / 100.0
                except (TypeError, ValueError):
                    return 0.0
            else:
                try:
                    num = float(raw)
                except (TypeError, ValueError):
                    return 0.0

        # If a model returns 5 for confidence, it usually means 0.5 rather than 0.05.
        if float(num).is_integer() and 1.0 <= num <= 9.0:
            num = num / 10.0
        # If a model returns 60 (meaning 60%), convert it.
        if 1.0 < num <= 100.0:
            num = num / 100.0
        return max(0.0, min(1.0, num))

    def merge_str_lists(existing, incoming) -> list[str]:
        left = coerce_to_str_list(existing)
        right = coerce_to_str_list(incoming)
        merged: list[str] = []
        seen: set[str] = set()
        for item in left + right:
            key = item.strip()
            if not key:
                continue
            lowered = key.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(key)
        return merged

    normalized: dict = {}
    for raw_key, value in payload.items():
        if not isinstance(raw_key, str):
            continue
        key = normalize_key(raw_key)
        key = aliases.get(key, key)
        if key not in known_fields:
            continue
        if key in normalized and not is_empty(normalized[key]):
            # Confidence is often emitted twice (e.g. 0.0 placeholder then a real value).
            if key == "confidence" and parse_confidence(value) > parse_confidence(normalized[key]):
                normalized[key] = value
            # LLMs frequently emit list-like fields multiple times under slightly different keys.
            # Merge instead of keeping only the first one.
            if key in {"findings", "impression", "limitations", "follow_up"}:
                normalized[key] = merge_str_lists(normalized[key], value)
            continue
        normalized[key] = value

    normalized["findings"] = coerce_to_str_list(normalized.get("findings"))
    normalized["impression"] = coerce_to_str_list(normalized.get("impression"))
    normalized["limitations"] = coerce_to_str_list(normalized.get("limitations"))
    normalized["follow_up"] = coerce_to_str_list(normalized.get("follow_up"))
    # Don't treat "adequate image quality" as a limitation; keep actual constraints like "single view".
    normalized["limitations"] = [
        item
        for item in normalized["limitations"]
        if "image quality" not in item.lower() or "adequate" not in item.lower()
    ]

    urgency = str(normalized.get("urgency", "")).strip().lower()
    normalized["urgency"] = urgency if urgency in {"routine", "expedited", "critical", "unknown"} else "unknown"

    normalized["confidence"] = parse_confidence(normalized.get("confidence", 0.0))

    for field in ("study_type", "clinical_context", "technique", "comparison"):
        if field in normalized and normalized[field] is not None:
            normalized[field] = str(normalized[field]).replace("\\_", "_").strip()

    return normalized


def extract_json_object(text: str) -> dict:
    candidate = extract_candidate_json_text(text)
    candidate = strip_json_line_comments(candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = repair_json(candidate)
        return json.loads(repaired)


def build_fallback_report(
    raw_text: str,
    study_type: str,
    clinical_context: str,
    comparison: str,
) -> StructuredReport:
    return StructuredReport(
        study_type=study_type or "Unknown study",
        clinical_context=clinical_context or "Not provided",
        technique="Not clearly identifiable from the single uploaded image.",
        comparison=comparison or "No comparison study provided.",
        findings=[],
        impression=["Unable to validate a structured JSON report from the model output."],
        urgency="unknown",
        confidence=0.0,
        limitations=[
            "The model response did not fully match the required report schema.",
            "A qualified clinician must review the raw output directly.",
        ],
        follow_up=["Repeat review by a radiologist or other qualified clinician."],
        raw_output=raw_text,
    )


def normalize_follow_up(study_type: str, follow_up: list[str]) -> list[str]:
    normalized = [item.strip() for item in follow_up if item.strip()]
    is_chest_xray = "chest x-ray" in study_type.lower() or "chest radiograph" in study_type.lower()
    if not is_chest_xray:
        return normalized

    guidance_present = any("6 week" in item.lower() or "6-week" in item.lower() for item in normalized)
    if guidance_present:
        return normalized
    return normalized


def should_add_default_cxr_follow_up(findings: list[str], impression: list[str]) -> bool:
    suspicious_terms = {
        "opacity",
        "consolidation",
        "pneumonia",
        "infiltrate",
        "airspace",
        "effusion",
        "pleural effusion",
    }
    text = " ".join([*findings, *impression]).lower()
    return any(term in text for term in suspicious_terms)


def derive_impression_from_findings(findings: list[str], impression: list[str]) -> list[str]:
    if impression:
        return impression
    text = " ".join(findings).lower()
    if not findings:
        return []
    if any(term in text for term in {"opacity", "consolidation", "pneumonia", "infiltrate", "airspace"}):
        for item in findings:
            lowered = item.lower()
            if any(term in lowered for term in {"opacity", "consolidation", "pneumonia", "infiltrate", "airspace"}):
                return [item.rstrip(".") + "."]
    if "effusion" in text:
        for item in findings:
            if "effusion" in item.lower():
                return [item.rstrip(".") + "."]
    if "pneumothorax" in text:
        for item in findings:
            if "pneumothorax" in item.lower():
                return [item.rstrip(".") + "."]
    return ["No acute cardiopulmonary abnormality identified."]


def derive_confidence(model_confidence: float, findings: list[str], impression: list[str]) -> float:
    text = " ".join([*findings, *impression]).lower()
    heuristic = 0.35
    abnormal_terms = {
        "opacity",
        "consolidation",
        "pneumonia",
        "infiltrate",
        "airspace",
        "effusion",
        "pneumothorax",
        "edema",
        "cardiomegaly",
    }
    uncertainty_terms = {"possible", "may", "suggestive", "likely", "cannot exclude"}
    if any(term in text for term in abnormal_terms):
        heuristic = 0.55
    if any(term in text for term in uncertainty_terms):
        heuristic = min(0.5, heuristic)
    if "no acute" in text:
        heuristic = 0.3
    return max(0.1, model_confidence, heuristic)


def parse_report_template(raw_text: str) -> dict:
    header_aliases = {
        "STUDY_TYPE": "study_type",
        "CLINICAL_CONTEXT": "clinical_context",
        "TECHNIQUE": "technique",
        "COMPARISON": "comparison",
        "FINDINGS": "findings",
        "IMPRESSION": "impression",
        "URGENCY": "urgency",
        "CONFIDENCE": "confidence",
        "LIMITATIONS": "limitations",
        "FOLLOW_UP": "follow_up",
        "FOLLOW-UP": "follow_up",
        "FOLLOW UP": "follow_up",
    }
    list_fields = {"findings", "impression", "limitations", "follow_up"}
    header_names = list(header_aliases.keys())
    header_pattern = "|".join(re.escape(f"{name}:") for name in header_names)

    def normalize_header_name(text: str) -> str:
        return re.sub(r"[\s\-]+", "_", text.strip().upper())

    normalized_header_names = {normalize_header_name(name) for name in header_names}

    def looks_like_partial_header(line: str) -> bool:
        candidate = normalize_header_name(line.rstrip(":"))
        if not candidate or not re.fullmatch(r"[A-Z_]+", candidate):
            return False
        return any(name.startswith(candidate) for name in normalized_header_names)

    normalized_text = re.sub(r"```(?:text|json)?", "", raw_text, flags=re.IGNORECASE)
    normalized_text = normalized_text.replace("```", "")
    normalized_text = re.sub(r"([A-Z][A-Z_ \-]+?)\s+:", r"\1:", normalized_text)
    # Models often collapse multiple sections onto one line. Split them back out.
    normalized_text = re.sub(rf"\s+({header_pattern})", r"\n\1", normalized_text)
    first_header_match = re.search(rf"(?m)^\s*({header_pattern})", normalized_text)
    if first_header_match:
        normalized_text = normalized_text[first_header_match.start():]
    parsed: dict = {}
    current_field = None

    for raw_line in normalized_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        matched_header = False
        for header, field_name in header_aliases.items():
            header_match = re.match(rf"^{re.escape(header)}\s*:\s*(.*)$", line, flags=re.IGNORECASE)
            if header_match:
                current_field = field_name
                remainder = header_match.group(1).strip()
                if field_name in list_fields:
                    parsed.setdefault(field_name, [])
                    if remainder:
                        parsed[field_name].append(remainder.lstrip("-* ").strip())
                else:
                    parsed[field_name] = remainder
                matched_header = True
                break

        if matched_header:
            continue

        if looks_like_partial_header(line):
            continue

        if current_field in list_fields:
            parsed.setdefault(current_field, [])
            parsed[current_field].append(line.lstrip("-* ").strip())
        elif current_field is not None and current_field not in parsed:
            parsed[current_field] = line

    if not parsed:
        raise ValueError("No template sections found in model output.")
    return parsed


def draft_report(
    bundle: LoadedMedGemma,
    image: Image.Image,
    study_type: str,
    clinical_context: str,
    comparison: str,
) -> StructuredReport:
    try:
        raw_parts: dict[str, str] = {}

        findings_raw = run_generation(
            bundle,
            image,
            build_section_prompt("findings", study_type, clinical_context, comparison),
            max_new_tokens=180,
        )
        raw_parts["findings"] = findings_raw
        findings = clean_list_items("findings", extract_bullet_list(findings_raw))

        impression_raw = run_generation(
            bundle,
            image,
            build_section_prompt("impression", study_type, clinical_context, comparison, findings=findings),
            max_new_tokens=100,
        )
        raw_parts["impression"] = impression_raw
        impression = derive_impression_from_findings(
            findings,
            clean_list_items("impression", extract_bullet_list(impression_raw)),
        )

        technique_raw = run_generation(
            bundle,
            image,
            build_section_prompt("technique", study_type, clinical_context, comparison),
            max_new_tokens=40,
        )
        raw_parts["technique"] = technique_raw
        technique = finalize_technique(study_type, technique_raw)

        urgency_raw = run_generation(
            bundle,
            image,
            build_section_prompt("urgency", study_type, clinical_context, comparison, findings=findings, impression=impression),
            max_new_tokens=12,
        )
        raw_parts["urgency"] = urgency_raw
        urgency = extract_urgency_value(urgency_raw)

        confidence_raw = run_generation(
            bundle,
            image,
            build_section_prompt("confidence", study_type, clinical_context, comparison, findings=findings, impression=impression),
            max_new_tokens=12,
        )
        raw_parts["confidence"] = confidence_raw
        confidence = derive_confidence(
            extract_confidence_value(confidence_raw),
            findings,
            impression,
        )

        limitations_raw = run_generation(
            bundle,
            image,
            build_section_prompt("limitations", study_type, clinical_context, comparison, findings=findings, impression=impression),
            max_new_tokens=80,
        )
        raw_parts["limitations"] = limitations_raw
        limitations = clean_list_items("limitations", extract_bullet_list(limitations_raw))

        follow_up_raw = run_generation(
            bundle,
            image,
            build_section_prompt("follow_up", study_type, clinical_context, comparison, findings=findings, impression=impression),
            max_new_tokens=100,
        )
        raw_parts["follow_up"] = follow_up_raw
        follow_up = clean_list_items("follow_up", extract_bullet_list(follow_up_raw))

        payload = normalize_report_payload(
            {
                "study_type": study_type or "Unknown study",
                "clinical_context": clinical_context or "Not provided",
                "technique": technique,
                "comparison": comparison or "No comparison study provided.",
                "findings": findings,
                "impression": impression,
                "urgency": urgency,
                "confidence": confidence,
                "limitations": limitations,
                "follow_up": follow_up,
            }
        )
        payload["follow_up"] = normalize_follow_up(
            payload.get("study_type", study_type or "Unknown study"),
            payload.get("follow_up", []),
        )
        if (
            "chest x-ray" in payload.get("study_type", study_type or "Unknown study").lower()
            and not payload["follow_up"]
            and should_add_default_cxr_follow_up(
                payload.get("findings", []),
                payload.get("impression", []),
            )
        ):
            payload["follow_up"] = [FOLLOW_UP_GUIDANCE]
        payload["raw_output"] = "\n\n".join(
            [
                "=== FINDINGS ===",
                raw_parts["findings"],
                "=== IMPRESSION ===",
                raw_parts["impression"],
                "=== TECHNIQUE ===",
                raw_parts["technique"],
                "=== URGENCY ===",
                raw_parts["urgency"],
                "=== CONFIDENCE ===",
                raw_parts["confidence"],
                "=== LIMITATIONS ===",
                raw_parts["limitations"],
                "=== FOLLOW_UP ===",
                raw_parts["follow_up"],
            ]
        )
        return StructuredReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError):
        raw_text = "\n\n".join(f"{key.upper()}:\n{value}" for key, value in raw_parts.items()) if "raw_parts" in locals() else ""
        return build_fallback_report(raw_text, study_type, clinical_context, comparison)
