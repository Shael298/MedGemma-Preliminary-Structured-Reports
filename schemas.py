from typing import Literal

from pydantic import BaseModel, Field


class StructuredReport(BaseModel):
    study_type: str = Field(description="Imaging study name or modality.")
    clinical_context: str = Field(
        description="Short summary of the clinical question or indication."
    )
    technique: str = Field(description="How the image was acquired, if known.")
    comparison: str = Field(
        default="No comparison study provided.",
        description="Relevant prior study comparison.",
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Observed imaging findings as short bullet-style statements.",
    )
    impression: list[str] = Field(
        default_factory=list,
        description="High-level summary points for clinician review.",
    )
    urgency: Literal["routine", "expedited", "critical", "unknown"] = "unknown"
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model self-rated confidence between 0 and 1.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Reasons the draft may be incomplete or unreliable.",
    )
    follow_up: list[str] = Field(
        default_factory=list,
        description="Suggested next review steps for a qualified clinician.",
    )
    disclaimer: str = Field(
        default=(
            "AI-generated preliminary draft for clinician review only. "
            "Not a final diagnostic report."
        )
    )
    raw_output: str = Field(
        default="",
        description="Original free-text model response kept for auditability.",
    )

    def to_markdown(self) -> str:
        def sentence(text: str) -> str:
            trimmed = (text or "").strip()
            if not trimmed:
                return trimmed
            if trimmed.endswith((".", "!", "?")):
                return trimmed
            return trimmed + "."

        findings_text = "\n".join(f"- {item}" for item in self.findings) or "- None provided"
        impression_text = "\n".join(f"- {item}" for item in self.impression) or "- None provided"
        limitation_text = "\n".join(f"- {item}" for item in self.limitations) or "- None provided"
        follow_up_text = "\n".join(f"- {item}" for item in self.follow_up) or "- None provided"

        return "\n".join(
            [
                f"**Study Type:** {sentence(self.study_type)}  ",
                f"**Clinical Context:** {sentence(self.clinical_context)}  ",
                f"**Technique:** {sentence(self.technique)}  ",
                f"**Comparison:** {sentence(self.comparison)}  ",
                f"**Urgency:** {sentence(self.urgency)}  ",
                f"**Confidence:** {self.confidence:.2f}.",
                "",
                "**Findings**",
                findings_text,
                "",
                "**Impression**",
                impression_text,
                "",
                "**Limitations**",
                limitation_text,
                "",
                "**Follow-up**",
                follow_up_text,
                "",
                f"**Disclaimer:** {self.disclaimer}",
            ]
        )
