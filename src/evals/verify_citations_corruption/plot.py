"""Plot the confirmed rate per variant from the eval's JSONL as horizontal bars."""

import matplotlib.pyplot as plt

from core import BASE_DIR
from evals.verify_citations_corruption.pipeline import (
    OUT_PATH,
    VARIANTS,
    _load_records,
    summarize,
)

IMAGE_PATH = BASE_DIR / "docs" / "verify_citations_corruption.png"

LABELS = {
    "real": "Original citations\n(should be confirmed)",
    "corrupt_claim_factual": "Claim with changed facts\n(should be rejected)",
    "corrupt_claim_complete": "Claim replaced by a distant false one\n(should be rejected)",
    "corrupt_verification_instructions": "Wrong verification instructions",
}
COLOR = "#4c72b0"


def plot() -> None:
    records = _load_records()
    if not records:
        raise SystemExit(f"no records in {OUT_PATH}; run verify_citations_corruption first")
    summary = summarize(records)

    names = list(VARIANTS)[::-1]
    rates = [100 * (summary[n]["confirmed_rate"] or 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh([LABELS[n] for n in names], rates, color=COLOR, height=0.6)
    for bar, n, rate in zip(bars, names, rates):
        s = summary[n]
        inside = rate > 50
        ax.text(
            rate - 1.5 if inside else rate + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%  ({s['confirmed']}/{s['citations']})",
            va="center",
            ha="right" if inside else "left",
            color="white" if inside else "black",
            fontsize=10,
        )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Citations the verifier confirmed (%)")
    ax.set_title(
        f"Citation verifier under corruption\n({len(records)} S3 answers, "
        f"{summary['real']['citations']} citations)",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_PATH, dpi=150, bbox_inches="tight")
    print(f"wrote {IMAGE_PATH}")


if __name__ == "__main__":
    plot()
