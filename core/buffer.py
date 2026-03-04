import asyncio
import time
from typing import List, Callable, Awaitable
from ..config import Config


class GroupChatBuffer:
    def __init__(self, group_id: str, flush_callback: Callable[[str, str, str], Awaitable[None]]):
        self.group_id = group_id
        self.messages: List[str] = []
        self.flush_callback = flush_callback
        self._timeout_task = None
        self._lock = asyncio.Lock()

    async def add_message(self, sender: str, content: str):
        if len(content) < Config.MSG_MIN_LENGTH and not any(k in content for k in ["?", "？", "怎么"]):
            return 

        current_time = time.strftime("%H:%M")
        msg_line = f"[{current_time}] {sender}: {content}"
        
        async with self._lock:
            self.messages.append(msg_line)
            
            if self._timeout_task:
                self._timeout_task.cancel()
                try:
                    await self._timeout_task
                except asyncio.CancelledError:
                    pass

            total_len = sum(len(m) for m in self.messages)
            if total_len >= Config.BUFFER_MAX_CHARS:
                await self._flush("buffer_full")
            else:
                self._timeout_task = asyncio.create_task(self._wait_and_flush())

    async def _wait_and_flush(self):
        try:
            await asyncio.sleep(Config.BUFFER_IDLE_TIMEOUT)
            async with self._lock:
                await self._flush("idle_timeout")
        except asyncio.CancelledError:
            pass

    async def _flush(self, reason: str):
        if not self.messages:
            return
        
        chunk_text = "\n".join(self.messages)
        self.messages = []
        
        await self.flush_callback(self.group_id, chunk_text, reason)
