#embedding_agent.py
import os
import asyncio
import time
from typing import List, Dict
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AzureOpenAI, RateLimitError, APIConnectionError, InternalServerError
from dotenv import load_dotenv
from tiktoken import get_encoding
from server.utils.logger import logger
import random

load_dotenv()

# class EmbeddingAgent:
#     def __init__(self):
#         self.client = AzureOpenAI(
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
#         )
#         self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
#         self.batch_size = 5  # Tune for performance vs. rate limits
#         self.cooldown = 3     # Cooldown in seconds between batches

#     def _log_sources(self, embedded: List[Dict]):
#         sources = {chunk.get("metadata", {}).get("source", "UNKNOWN") for chunk in embedded}
#         logger.info((f"\n-------------Ashish Testing metadata starts here"))
#         logger.info(f"Available sources in metadata: {sources}")

#     def _log_token_usage(self, embedded: List[Dict]):
#         # print("[DEBUG] EmbeddingAgent._log_token_usage executed, reading metadata.")

#         total_tokens = sum(chunk.get("tokens", 0) for chunk in embedded)
#         kb_tokens = sum(chunk.get("tokens", 0) for chunk in embedded if chunk.get("metadata", {}).get("source") == "knowledge_bank")
#         fi_tokens = sum(chunk.get("tokens", 0) for chunk in embedded if chunk.get("metadata", {}).get("source") == "field_issues")
#         prd_tokens = sum(chunk.get("tokens", 0) for chunk in embedded if chunk.get("metadata", {}).get("source") == "prds")

#     async def _embed_batch_with_retry(self, texts: List[str], max_retries: int = 5):
#         """
#         Async Azure embeddings call with exponential backoff on 429.
#         """
#         for attempt in range(max_retries):
#             try:
#                 return await asyncio.to_thread(
#                     self.client.embeddings.create,
#                     input=texts,
#                     model=self.deployment
#                 )
#             except Exception as e:
#                 if "429" in str(e) or "RateLimitError" in str(type(e)):
#                     wait = (2 ** attempt) + random.uniform(0, 1)
#                     logger.warning(f"[EmbeddingAgent] 429 received, retrying in {wait:.2f}s...")
#                     await asyncio.sleep(wait)
#                 else:
#                     raise
#         raise RuntimeError("[EmbeddingAgent] Max retries exceeded for batch")
    
#     ##################################################################################################
#     # async def embed_chunks_async(self, chunks: List[Dict]) -> List[Dict]:
#     #     embedded_chunks = []
#     #     i = 0
#     #     retries = 0
#     #     max_retries = 5

#     #     while i < len(chunks):
#     #         batch = chunks[i:i + self.batch_size]
#     #         batch_texts = [chunk["text"] for chunk in batch]

#     #         try:
#     #             response = await self._embed_batch_with_retry(batch_texts)
#     #             for idx, item in enumerate(response.data):
#     #                 # embedded_chunks.append({
#     #                 #     "text": batch[idx]["text"],
#     #                 #     "embedding": item.embedding,
#     #                 #     "metadata": batch[idx].get("metadata", {}),
#     #                 #     # 🚨 remove token count from here later (move to chunker)
#     #                 #     "tokens": self._count_tokens(batch[idx]["text"])
#     #                 # })
#     #                 embedded_chunks.append({
#     #                     "text": batch[idx]["text"],
#     #                     "embedding": item.embedding,
#     #                     "metadata": batch[idx].get("metadata", {})
#     #                 })

#     #             i += self.batch_size
#     #             retries = 0  # reset retries
#     #             await asyncio.sleep(self.cooldown)

#     #         except Exception as e:
#     #             if "429" in str(e) and retries < max_retries:
#     #                 wait_time = min(self.cooldown * (2 ** retries), 10)  # cap at 10s
#     #                 logger.warning(f"[EmbeddingAgent] ⚠️ 429 rate limit. Retry {retries+1}/{max_retries}, waiting {wait_time}s")
#     #                 retries += 1
#     #                 await asyncio.sleep(wait_time)
#     #             else:
#     #                 logger.error(f"[EmbeddingAgent] ❌ Batch failed: {type(e).__name__}: {e}")
#     #                 i += self.batch_size  # skip this batch and continue

#     #     # self._log_token_usage(embedded_chunks)
#     #     return embedded_chunks
#     ##########################################################################################################################
#     async def embed_chunks_async(self, chunks: List[Dict]) -> List[Dict]:
#         """Embed chunks using parallel async batches with smart 429 handling."""
#         embedded_chunks = []
#         batch_size = self.batch_size

#         # Split into batches
#         batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

#         async def process_batch(batch):
#             batch_texts = [c["text"] for c in batch]

#             async def try_embed(batch_texts, size):
#                 try:
#                     response = await self._embed_batch_with_retry(batch_texts)
#                     results = []
#                     for idx, item in enumerate(response.data):
#                         results.append({
#                             "text": batch[idx]["text"],
#                             "embedding": item.embedding,
#                             "metadata": batch[idx].get("metadata", {})
#                         })
#                     return results
#                 except Exception as e:
#                     if "429" in str(e) and size > 5:
#                         mid = len(batch_texts) // 2
#                         logger.warning(f"[EmbeddingAgent] ⚠️ 429: splitting batch of {size} → {mid}+{len(batch_texts)-mid}")
#                         left = await try_embed(batch_texts[:mid], mid)
#                         right = await try_embed(batch_texts[mid:], len(batch_texts) - mid)
#                         return left + right
#                     else:
#                         logger.error(f"[EmbeddingAgent] Batch failed: {type(e).__name__}: {e}")
#                         return []

#             return await try_embed(batch_texts, len(batch_texts))

#         # Run batches in parallel (limit concurrency to avoid API overload)
#         semaphore = asyncio.Semaphore(1)  # tune up/down for speed vs. safety

#         async def sem_task(batch):
#             async with semaphore:
#                 return await process_batch(batch)

#         tasks = [sem_task(b) for b in batches]
#         results = await asyncio.gather(*tasks)

#         # Flatten results
#         for r in results:
#             embedded_chunks.extend(r)

#         # logger.info(f"[EmbeddingAgent] ✅ Completed embeddings: {len(embedded_chunks)} total")
#         return embedded_chunks



#     # async def embed_chunks_async(self, chunks: List[Dict]) -> List[Dict]:
#     #     #logger.info(f"\n [EmbeddingAgent] 🚀 Embedding {len(chunks)} chunks with async dynamic batching...")

#     #     embedded_chunks = []
#     #     current_batch_size = self.batch_size
#     #     current_cooldown = self.cooldown
#     #     i = 0

#     #     while i < len(chunks):
#     #         batch = chunks[i:i + current_batch_size]
#     #         batch_texts = [chunk["text"] for chunk in batch]

#     #         try:
#     #             # response = await asyncio.to_thread(self._embed_batch_with_retry, batch_texts)
#     #             response = await self._embed_batch_with_retry(batch_texts)
#     #             for idx, item in enumerate(response.data):
#     #                 embedded_chunks.append({
#     #                     "text": batch[idx]["text"],
#     #                     "embedding": item.embedding,
#     #                     "metadata": batch[idx].get("metadata", {}),
#     #                     "tokens": self._count_tokens(batch[idx]["text"])
#     #                 })
#     #             # if embedded_chunks:
#     #             #     logger.info("------Ashish testing debugging starts here----------\n")
#     #             #     logger.info(f"[DEBUG] Sample post-embedding chunk: {embedded_chunks[0]}")
#     #             #     logger.info(f"[DEBUG] Vector length: {len(embedded_chunks[0]['embedding'])}")
#     #             #     logger.info(f"[DEBUG] Tokens: {embedded_chunks[0]['tokens']}")
#     #             #     logger.info("------Ashish testing debugging ends here----------\n")
#     #             i += current_batch_size
#     #             await asyncio.sleep(current_cooldown)

#     #         except Exception as e:
#     #             if "429" in str(e):
#     #                 if current_batch_size > 5:
#     #                     current_batch_size = max(5, current_batch_size // 2)
#     #                 current_cooldown = min(current_cooldown * 2, 30)
#     #                 logger.warning(f"[EmbeddingAgent] ⚠️ 429: shrinking batch → {current_batch_size}, cooldown → {current_cooldown}s")
#     #                 await asyncio.sleep(current_cooldown)
#     #             else:
#     #                 logger.error(f"[EmbeddingAgent] Batch failed: {type(e).__name__}: {e}")
#     #                 i += current_batch_size  # skip batch to continue

#     #     self._log_token_usage(embedded_chunks)
#     #     return embedded_chunks


#     def _count_tokens(self, text: str) -> int:
#         tokenizer = get_encoding("cl100k_base")
#         return len(tokenizer.encode(text))


class EmbeddingAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        )
        self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        # Tunable
        self.batch_size = 20         # start small, safer
        self.cooldown = 2            # initial cooldown
        self.concurrency = 3         # how many batches run in parallel
        self.max_retries = 5

    async def _embed_batch_with_retry(self, texts: List[str]) -> List[Dict]:
        """Embed one batch with retry + exponential backoff."""
        delay = self.cooldown
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    input=texts,
                    model=self.deployment
                )
                return response
            except Exception as e:
                if "429" in str(e) or "RateLimitError" in str(type(e)):
                    logger.warning(f"[EmbeddingAgent] ⚠️ 429: attempt {attempt+1}/{self.max_retries}, retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay + random.uniform(0, 1))
                    delay = min(delay * 2, 30)  # exponential backoff, max 30s
                else:
                    logger.error(f"[EmbeddingAgent] ❌ Non-429 error: {type(e).__name__}: {e}")
                    raise
        logger.error("[EmbeddingAgent] ❌ Max retries exceeded for batch")
        return None
    async def embed_chunks_async(self, chunks: List[Dict]) -> List[Dict]:
        """Embed all chunks with safe concurrency + retry handling."""
        embedded_chunks = []
        batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]

        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_batch(batch, idx):
            async with semaphore:
                texts = [c["text"] for c in batch]
                response = await self._embed_batch_with_retry(texts)
                if not response:
                    return []
                results = []
                for i, item in enumerate(response.data):
                    # ✅ Ensure metadata always has a source
                    meta = batch[i].get("metadata", {})
                    if "source" not in meta:
                        meta["source"] = "unknown"

                    results.append({
                        "text": batch[i]["text"],
                        "embedding": item.embedding,
                        "metadata": meta,
                        "tokens": self._count_tokens(batch[i]["text"])
                    })
                logger.info(f"[EmbeddingAgent] ✅ Batch {idx+1}/{len(batches)} done ({len(batch)} chunks)")
                return results

        tasks = [process_batch(batch, idx) for idx, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)

        # Flatten
        for r in results:
            embedded_chunks.extend(r)

        logger.info(f"[EmbeddingAgent] 🎯 Completed embeddings: {len(embedded_chunks)}/{len(chunks)} chunks")
        return embedded_chunks

    # async def embed_chunks_async(self, chunks: List[Dict]) -> List[Dict]:
    #     """Embed all chunks with safe concurrency + retry handling."""
    #     embedded_chunks = []
    #     batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]

    #     semaphore = asyncio.Semaphore(self.concurrency)

    #     async def process_batch(batch, idx):
    #         async with semaphore:
    #             texts = [c["text"] for c in batch]
    #             response = await self._embed_batch_with_retry(texts)
    #             if not response:
    #                 return []
    #             results = []
    #             for i, item in enumerate(response.data):
    #                 results.append({
    #                     "text": batch[i]["text"],
    #                     "embedding": item.embedding,
    #                     "metadata": batch[i].get("metadata", {}),
    #                     "tokens": self._count_tokens(batch[i]["text"])
    #                 })
    #             logger.info(f"[EmbeddingAgent] ✅ Batch {idx+1}/{len(batches)} done ({len(batch)} chunks)")
    #             return results

        tasks = [process_batch(batch, idx) for idx, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)

        # Flatten
        for r in results:
            embedded_chunks.extend(r)

        logger.info(f"[EmbeddingAgent] 🎯 Completed embeddings: {len(embedded_chunks)}/{len(chunks)} chunks")
        return embedded_chunks

    def _count_tokens(self, text: str) -> int:
        tokenizer = get_encoding("cl100k_base")
        return len(tokenizer.encode(text))
