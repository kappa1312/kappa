"""
Kappa OS Interactive Chat Interface.

Provides a conversational interface for guiding users from ideation to project completion.
"""

from src.chat.interface import ConversationPhase, KappaChat, Message, start_chat_cli

__all__ = ["KappaChat", "ConversationPhase", "Message", "start_chat_cli"]
