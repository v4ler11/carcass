from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from agent.tools.plu_graph_tools.plu_graph import PLU_GRAPH_DIR, read_files
from chat.tools import Tool, ToolContext


class ReadPlanDeZonageCentralMeta(Tool):
    @property
    def name(self) -> str:
        return "read_plan_de_zonage_central_meta"

    async def execute(
        self,
        ctx: ToolContext,
        tool_call: ChatCompletionMessageToolCall,
        args: dict[str, Any],
    ) -> list[ChatCompletionMessageParam]:
        directory = PLU_GRAPH_DIR / "document" / "parsed" / "s2_ocr"
        return read_files(tool_call.id, directory)

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read and attach: 01_legende.png (general legend for zone "
                    "boundaries, protected/natural areas, risks, OAP sectors, "
                    "and ER symbols); 02_elements_recenses.png (map and notice "
                    "for elements listed under Code de l'urbanisme articles "
                    "L 151-19 and L 151-23, plan n°5.5); 03_zones_legend.png "
                    "(zoning color key for UA/UC/UD/UE/UG/UH/UM/UMG/UR, "
                    "UB/AURc, UL/UME, UN/UX, and A/N/Nc/NE/NF/NL); "
                    "04_er_elargissements_voirie.md (ER table for road "
                    "widening); 05_er_creation_amelioration_voirie.md (ER table "
                    "for road creation/improvement); "
                    "06_er_equipements_espaces_publics.md (ER table for public "
                    "equipment/spaces)."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        }
