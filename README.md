# Gemma Chest X-ray Reporter

Local Streamlit prototype for generating **preliminary chest X-ray report drafts** with **MedGemma 1.5**.

This is not a diagnostic product. I built it as a project about making multimodal model output structurally usable enough for a practical prototype.

## 1) What the current narrow implementation is

- Scope is **chest X-ray only**
- Tested upload formats are **PNG, JPG, and JPEG**
- The app generates a structured draft with:
  - findings
  - impression
  - urgency
  - confidence
  - limitations
  - follow-up
- MedGemma is loaded locally with 4-bit quantization when CUDA is available
- Python assembles the final report structure instead of trusting one large model response end-to-end

Current high-level flow:

1. Upload a chest X-ray
2. Run MedGemma section by section
3. Clean the outputs in Python
4. Assemble the final structured report
5. Render the result in Streamlit

## 2) Main learning

- **Quantization matters**: loading MedGemma in 4-bit on my laptop GPU was the practical way to make local inference usable at all.
- **One-shot JSON was fragile**: the model often misspelled keys, invented keys, emitted comments, or produced multiple JSON objects.
- **Regex/repair logic helps but is not enough**: extracting JSON-looking text, stripping comments, repairing malformed JSON, and normalizing keys improved robustness, but did not fully solve the problem.
- **Prompting alone is not enough**: for sections like follow-up, the model still drifted into findings or boilerplate unless the code also validated the output shape.
- **Smaller generations are easier to control**: generating one whole report was too unstable; generating one section at a time was much easier to clean and assemble.
- **Section-specific cleanup is important**: impressions needed different filtering from findings; follow-up needed recommendation-only filtering; confidence needed a code-level backstop when the model returned `0.0`.
- **Narrow scope helps a lot**: keeping the active app to chest X-ray only made the project much more stable and understandable.

## 3) Different implementations I tried and why I chose the current one

### A. One-shot JSON report

Idea:

- ask the model for one exact JSON object

Problem:

- too brittle in practice

### B. One-shot plain-text template

Idea:

- ask for a rigid template like `FINDINGS:`, `IMPRESSION:`, `FOLLOW_UP:`

Problem:

- better than JSON, but still prone to section drift, truncation, and planning text

### C. Full constrained decoding / grammar-style output

Idea:

- hard-constrain the model so it can only emit valid structured output

Problem:

- stronger structurally, but more engineering complexity than was needed for this prototype

### D. Current implementation: section-by-section generation plus Python assembly

Idea:

- ask MedGemma for findings, impression, confidence, follow-up, etc. separately
- clean each section with simple rules
- assemble the final report in code

Why I ended up choosing this:

- more robust than one-shot JSON
- simpler than full constrained decoding
- easier to debug
- keeps the final report structure under program control

In short:

**The main decision I made was to stop treating the model like a perfect serializer and let Python own the final structure.**

## Setup

1. Create and activate a virtual environment
2. If you want GPU inference on an Nvidia machine, install a CUDA-enabled PyTorch build first. A plain `pip install` may pull CPU-only `torch` on Windows.
3. Install dependencies with `pip install -e .`
4. Copy `.env.example` to `.env`
5. Set `HF_TOKEN`
6. Optionally set `MEDGEMMA_MODEL_ID`  
   Default: `google/medgemma-1.5-4b-it`
7. Run `python smoke_test.py`
8. Run `streamlit run app.py`

## Safety

This app generates draft outputs for clinician review only. It is not a final diagnostic system and should not be used as the sole basis for medical decisions.
