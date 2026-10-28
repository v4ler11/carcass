"""Filesystem locations for agent stage inputs and outputs."""

from core import BASE_DIR

S1_RESULTS_PATH = BASE_DIR / "out" / "s1" / "results.json"
S2_RESULTS_PATH = BASE_DIR / "out" / "s2" / "results.json"
S3_RESULTS_DIR = BASE_DIR / "out" / "s3" / "results"
S4_RESULTS_PATH = BASE_DIR / "out" / "s4" / "results.json"
S4_ANSWER_STORE_PATH = BASE_DIR / "out" / "s4" / "answer_store.json"
S4_MARKDOWN_PATH = BASE_DIR / "out" / "s4" / "result.md"
