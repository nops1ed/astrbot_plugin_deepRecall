from typing import Dict
from astrbot.api.all import *

from .config import Config
from .core.buffer import GroupChatBuffer
from .core.engine import RAGEngine


@register("rag_memory", "YourName", "High-concurrency group chat knowledge extraction and dual-path RAG plugin", "2.0.0")
class RAGMemoryPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        
        plugin_config = self.get_config()
        if plugin_config:
            self.config = Config.from_dict(plugin_config)
        else:
            self.config = Config.from_env()
            
        self.buffers: Dict[str, GroupChatBuffer] = {}
        self.rag_engine = RAGEngine(config=self.config)
        
    async def _on_buffer_flush(self, group_id: str, chunk_text: str, reason: str):
        chunk_id, topic = await self.rag_engine.ingest(group_id, chunk_text)
        print(f"[RAGMemory] ingested chunk={chunk_id}, topic={topic}, reason={reason}")

    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        msg_str = event.message_str
        group_id = event.message_obj.group_id
        sender_name = event.message_obj.sender.nickname
        
        if msg_str.startswith("/"):
            return

        if group_id not in self.buffers:
            self.buffers[group_id] = GroupChatBuffer(group_id, self._on_buffer_flush)
            
        await self.buffers[group_id].add_message(sender_name, msg_str)

    @filter.command("rag search")
    async def search_chat(self, event: AstrMessageEvent, *, query: str):
        group_id = event.message_obj.group_id
        yield event.plain_result(f"Searching: '{query}'...")
        
        retrieved_contexts = await self.rag_engine.search(group_id, query)
        
        if not retrieved_contexts:
            yield event.plain_result("No relevant memories found.")
            return
            
        context_str = "\n\n---\n\n".join(retrieved_contexts)
        prompt = f"""As a group chat memory assistant, answer questions based on the following real history snippets.
Do not fabricate information. If no information exists, say so directly.

[Query]: {query}
[Chat Snippets]:
{context_str}
"""
        response = await self.context.get_llm_provider().get_provider().text_chat(prompt)
        yield event.plain_result(f"Summary:\n{response.completion_text}")

    @filter.command("rag stats")
    async def show_stats(self, event: AstrMessageEvent):
        stats = self.rag_engine.get_stats()
        
        result = f"""RAG Memory Stats

Total docs: {stats['total_documents']}
Groups: {stats['total_groups']}
FAISS index: {stats['faiss_index_size']}
BM25 corpus: {stats['bm25_corpus_size']}
Index type: {stats['index_type']}

OK
"""
        yield event.plain_result(result)

    @filter.command("rag backup")
    async def backup(self, event: AstrMessageEvent):
        yield event.plain_result("Creating backup...")
        
        backup_path = self.rag_engine.create_backup()
        
        if backup_path:
            yield event.plain_result(f"Backup created: {backup_path}")
        else:
            yield event.plain_result("Backup failed, check logs")

    @filter.command("rag rebuild")
    async def rebuild_index(self, event: AstrMessageEvent):
        yield event.plain_result("Rebuilding index...")
        
        self.rag_engine.rebuild_index()
        
        stats = self.rag_engine.get_stats()
        yield event.plain_result(f"Index rebuilt, current docs: {stats['total_documents']}")

    @filter.command("rag forget")
    async def forget_old(self, event: AstrMessageEvent, *, days: str = None):
        try:
            days_int = int(days) if days else None
        except ValueError:
            days_int = None
        
        target_days = days_int or self.config.FORGET_DAYS
        yield event.plain_result(f"Forgetting memories older than {target_days} days...")
        
        deleted_count = self.rag_engine.forget_old_documents(target_days)
        
        if deleted_count > 0:
            yield event.plain_result(f"Forgot {deleted_count} old memories")
        else:
            yield event.plain_result("No old memories to forget")

    @filter.command("rag delete")
    async def delete_doc(self, event: AstrMessageEvent, *, doc_id: str):
        yield event.plain_result(f"Deleting memory {doc_id}...")
        
        success = self.rag_engine.delete_document(doc_id)
        
        if success:
            yield event.plain_result(f"Deleted memory {doc_id}")
        else:
            yield event.plain_result(f"Memory {doc_id} not found")

    @filter.command("rag help")
    async def show_help(self, event: AstrMessageEvent):
        help_text = """RAG Memory Plugin Help

Search commands:
/rag search <query> - Search group chat memories

Admin commands:
/rag stats - View knowledge base stats
/rag backup - Create backup
/rag rebuild - Rebuild index
/rag forget [days] - Forget old memories (default 30 days)
/rag delete <id> - Delete specific memory
/rag help - Show help

Config:
Use environment variables, see .env.example
"""
        yield event.plain_result(help_text)
