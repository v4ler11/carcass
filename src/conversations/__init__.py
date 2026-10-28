"""Read side of the traces: conversations, their metadata and their span trees.

Nothing here touches the content blobs until a node asks for them::

    from conversations import get_conversation, list_conversations

    for conversation in list_conversations():          # trees not loaded
        print(conversation.conversation_id, conversation.turns, conversation.input_tokens)

    conversation = get_conversation(conversation.conversation_id)   # populated
    for turn in conversation.tree:
        turn.populate()      # loads the refs of the turn and every node below it
        print(turn.name, turn.duration_ms, [child.name for child in turn.children])
        for child in turn.children:
            if child.name == "chat":
                print(child.attributes["gen_ai.input.messages"])

A listed conversation loads its tree when `populate()` is called; reading
`conversation.tree` before that raises.
"""

from conversations.conversation import Conversation
from conversations.node import Node
from conversations.reader import get_conversation, list_conversations

__all__ = [
    "Conversation",
    "Node",
    "get_conversation",
    "list_conversations",
]
