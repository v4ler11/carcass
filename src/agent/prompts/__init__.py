from agent.prompts.quick_citation import CITATION_PROMPT
from agent.prompts.s1_overlay_resolution import (
    SYSTEM_PROMPT as S1_OVERLAY_RESOLUTION_SYSTEM_PROMPT,
)
from agent.prompts.s1_overlay_resolution import (
    USER_PROMPT as S1_OVERLAY_RESOLUTION_USER_PROMPT,
)
from agent.prompts.s2_fan_out_questions import (
    USER_PROMPT as S2_FAN_OUT_QUESTIONS_USER_PROMPT,
)
from agent.prompts.s2_fan_out_questions import (
    render_system_prompt as render_s2_fan_out_questions_system_prompt,
)
from agent.prompts.s3_qna import (
    render_question_input as render_s3_qna_question_input,
)
from agent.prompts.s3_qna import (
    render_system_prompt as render_s3_qna_system_prompt,
)

__all__ = [
    "CITATION_PROMPT",
    "S1_OVERLAY_RESOLUTION_SYSTEM_PROMPT",
    "S1_OVERLAY_RESOLUTION_USER_PROMPT",
    "S2_FAN_OUT_QUESTIONS_USER_PROMPT",
    "render_s2_fan_out_questions_system_prompt",
    "render_s3_qna_question_input",
    "render_s3_qna_system_prompt",
]
