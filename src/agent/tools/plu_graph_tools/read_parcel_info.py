from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from agent.tools.plu_graph_tools.plu_graph import PLU_GRAPH_DIR, read_files
from chat.tools import Tool, ToolContext


class ReadParcelInfo(Tool):
    @property
    def name(self) -> str:
        return "read_parcel_info"

    async def execute(
        self,
        ctx: ToolContext,
        tool_call: ChatCompletionMessageToolCall,
        args: dict[str, Any],
    ) -> list[ChatCompletionMessageParam]:
        return read_files(
            tool_call.id,
            PLU_GRAPH_DIR / "parcel",
            skip={"parcel.pdf", "zoom-0.png"},
        )

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read and attach: metadata_parcel.json (target parcel "
                    "identity, GPU zone/prescriptions/servitudes, georeference "
                    "residuals, and zoom index); parcel.wkt (exact parcel "
                    "boundary in Lambert-93); zoom-1.png (~2.5 km context "
                    "window); zoom-2.png (~864 m window); zoom-3.png (~381 m "
                    "tightest window around the parcel)."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        }
