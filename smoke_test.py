from dotenv import load_dotenv
import os

from medgemma_service import (
    format_diagnostics,
    get_runtime_diagnostics,
    load_medgemma,
)

load_dotenv()


def main() -> None:
    model_id = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
    hf_token = os.getenv("HF_TOKEN")

    diagnostics = get_runtime_diagnostics()
    print("Runtime diagnostics:")
    print(format_diagnostics(diagnostics))
    print()
    print(f"Attempting to load model: {model_id}")

    bundle = load_medgemma(model_id=model_id, hf_token=hf_token)
    print("Model load succeeded.")
    print(f"device: {bundle.device}")
    print(f"quantized_4bit: {bundle.quantized_4bit}")
    print(f"compute_dtype: {bundle.compute_dtype}")


if __name__ == "__main__":
    main()