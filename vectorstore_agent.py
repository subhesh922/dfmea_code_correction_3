import os
import uuid
import math
from typing import List, Dict
from dotenv import load_dotenv
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from server.utils.logger import logger
import time 
import random 

load_dotenv()

# class VectorStoreAgent:
#     def __init__(self, collection_name: str = None):
#         self.qdrant_url = os.getenv("QDRANT_ENDPOINT")
#         self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
#         # Always use fixed name unless explicitly overridden
#         self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "dfmea_collection")

#         self.client = QdrantClient(
#             url=self.qdrant_url,
#             api_key=self.qdrant_api_key,
#             prefer_grpc=False,  # Make HTTP-based async fallback smoother
#             https=True,
#             timeout=120,  # increased from 30s to 120s for stability
#             verify=False
#         )
#         # Disable SSL verification for all requests
#         self.ssl_verify = False

#     def create_collection(self, vector_dim: int):
#         print(f"[VectorStoreAgent] Creating collection '{self.collection_name}'...")
#         self.client.recreate_collection(
#             collection_name=self.collection_name,
#             vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
#         )
#         print(f"[VectorStoreAgent] ✅ Collection ready: {self.collection_name}")

#     def add_embeddings(self, embedded_chunks: List[Dict], batch_limit: int = 100):
#         print(f"[VectorStoreAgent] Uploading {len(embedded_chunks)} vectors in batches...")

#         points = [
#             PointStruct(
#                 id=idx,
#                 vector=chunk["embedding"],
#                 payload={**chunk.get("metadata", {}), "text": chunk["text"]}
#             )
#             for idx, chunk in enumerate(embedded_chunks)
#         ]

#         num_batches = math.ceil(len(points) / batch_limit)
#         for i in range(num_batches):
#             batch = points[i * batch_limit : (i + 1) * batch_limit]
#             try:
#                 self.client.upsert(collection_name=self.collection_name, points=batch)
#             except Exception as e:
#                 print(f"[VectorStoreAgent] Batch {i+1} failed: {type(e).__name__}: {e}")
#         print("[VectorStoreAgent] ✅ Upload complete.")

#     def search(self, query: str, top_k: int = 5) -> List[Dict]:
#         print(f"[VectorStoreAgent] Searching for: '{query}' in '{self.collection_name}'")

#         embedding_client = AzureOpenAI(
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
#         )
#         deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
#         response = embedding_client.embeddings.create(input=query, model=deployment)
#         query_vector = response.data[0].embedding

#         results = self.client.search(
#             collection_name=self.collection_name,
#             query_vector=query_vector,
#             limit=top_k,
#             with_payload=True
#         )

#         output = []
#         for hit in results:
#             output.append({
#                 "score": hit.score,
#                 "text": hit.payload.get("text", ""),
#                 "metadata": hit.payload
#             })

#         print(f"[VectorStoreAgent] Found {len(output)} matches.")
#         return output

#     def delete_collection(self):
#         print(f"[VectorStoreAgent] Dropping collection '{self.collection_name}'...")
#         self.client.delete_collection(self.collection_name)
#         print("[VectorStoreAgent] Collection deleted.")


class VectorStoreAgent:
    def __init__(self, collection_name: str = None):
        self.qdrant_url = os.getenv("QDRANT_ENDPOINT")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        # Always use fixed name unless explicitly overridden
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "dfmea_collection")

        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=False,  # Make HTTP-based async fallback smoother
            https=True,
            timeout=120,  # increased from 30s to 120s for stability
            verify=False
        )
        self.ssl_verify = False

    def create_collection(self, vector_dim: int):
        logger.info(f"[VectorStoreAgent] Creating collection '{self.collection_name}'...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        logger.info(f"[VectorStoreAgent] ✅ Collection ready: {self.collection_name}")

    # def add_embeddings(self, embedded_chunks: List[Dict], batch_limit: int = 100):
    #     total = len(embedded_chunks)
    #     points = [
    #         PointStruct(
    #             id=idx,
    #             vector=chunk["embedding"],
    #             payload={**chunk.get("metadata", {}), "text": chunk["text"]}
    #         )
    #         for idx, chunk in enumerate(embedded_chunks)
    #     ]

    #     num_batches = math.ceil(total / batch_limit)
    #     logger.info(f"[VectorStoreAgent] 🚀 Uploading {total} vectors in {num_batches} batches...")

    #     for i in range(num_batches):
    #         batch = points[i * batch_limit : (i + 1) * batch_limit]
    #         try:
    #             self.client.upsert(collection_name=self.collection_name, points=batch)
    #             percent = ((i+1) / num_batches) * 100
    #             logger.info(f"[VectorStoreAgent] 📊 Progress: {percent:.0f}% ({i+1}/{num_batches} batches done)")
    #         except Exception as e:
    #             logger.error(f"[VectorStoreAgent] ❌ Batch {i+1}/{num_batches} failed: {type(e).__name__}: {e}")

    #     logger.info(f"[VectorStoreAgent] 🎯 Upload complete: {total} vectors stored in {self.collection_name}")
    def add_embeddings(self, embedded_chunks: List[Dict], batch_limit: int = 100):
        total = len(embedded_chunks)
        points = []

        for idx, chunk in enumerate(embedded_chunks):
            metadata = chunk.get("metadata", {})

            # ✅ Ensure core metadata fields are always present
            payload = {
                "source": metadata.get("source", "unknown"),          # prds / knowledge_base / field_issues
                "product": metadata.get("product", "unspecified"),
                "subproduct": metadata.get("subproduct", "unspecified"),
                "tokens": chunk.get("tokens", 0),
                "text": chunk["text"],                               # keep full text for context
            }

            points.append(
                PointStruct(
                    id=idx,
                    vector=chunk["embedding"],
                    payload=payload
                )
            )

        num_batches = math.ceil(total / batch_limit)
        logger.info(f"[VectorStoreAgent] 🚀 Uploading {total} vectors in {num_batches} batches...")

        for i in range(num_batches):
            batch = points[i * batch_limit : (i + 1) * batch_limit]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                percent = ((i+1) / num_batches) * 100
                logger.info(f"[VectorStoreAgent] 📊 Progress: {percent:.0f}% ({i+1}/{num_batches} batches done)")
            except Exception as e:
                logger.error(f"[VectorStoreAgent] ❌ Batch {i+1}/{num_batches} failed: {type(e).__name__}: {e}")

        logger.info(f"[VectorStoreAgent] 🎯 Upload complete: {total} vectors stored in {self.collection_name}")


    def _embed_query_with_retry(self, embedding_client, query: str, deployment: str, max_retries: int = 5, cooldown: int = 2):
        """Get embedding for query with retry + exponential backoff on 429 errors."""
        delay = cooldown
        for attempt in range(max_retries):
            try:
                response = embedding_client.embeddings.create(input=query, model=deployment)
                return response
            except Exception as e:
                if "429" in str(e) or "RateLimitError" in str(type(e)):
                    logger.warning(f"[VectorStoreAgent] ⚠️ 429 during query embed (attempt {attempt+1}/{max_retries}), retrying in {delay:.2f}s...")
                    time.sleep(delay + random.uniform(0, 1))
                    delay = min(delay * 2, 30)  # exponential backoff, max 30s
                else:
                    logger.error(f"[VectorStoreAgent] ❌ Non-429 error during query embed: {type(e).__name__}: {e}")
                    raise
        logger.error("[VectorStoreAgent] ❌ Max retries exceeded for query embedding")
        return None

    # def search(self, query: str, top_k: int = 5) -> List[Dict]:
    #     logger.info(f"[VectorStoreAgent] 🔎 Searching for: '{query}' in '{self.collection_name}'")

    #     embedding_client = AzureOpenAI(
    #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #         api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
    #     )
    #     deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    #     response = self._embed_query_with_retry(embedding_client, query, deployment)
    #     if not response:
    #         logger.error("[VectorStoreAgent] ❌ Failed to embed query after retries")
    #         return []

    #     query_vector = response.data[0].embedding

    #     results = self.client.search(
    #         collection_name=self.collection_name,
    #         query_vector=query_vector,
    #         limit=top_k,
    #         with_payload=True
    #     )

    #     output = []
    #     for idx, hit in enumerate(results, 1):
    #         output.append({
    #             "score": hit.score,
    #             "text": hit.payload.get("text", ""),
    #             "metadata": hit.payload
    #         })
    #         logger.info(f"[VectorStoreAgent]   ↳ Result {idx}: score={hit.score:.4f}, preview='{hit.payload.get('text','')[:50]}...'")

    #     logger.info(f"[VectorStoreAgent] 🎯 Found {len(output)} matches")
    #     return output
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        logger.info(f"[VectorStoreAgent] 🔎 Searching for: '{query}' in '{self.collection_name}'")

        embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        )
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        # Retry loop for 429 errors
        delay = 2
        for attempt in range(5):
            try:
                response = embedding_client.embeddings.create(input=query, model=deployment)
                break
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"[VectorStoreAgent] ⚠️ 429 during search embed: attempt {attempt+1}/5, retrying in {delay}s...")
                    time.sleep(delay + random.uniform(0, 1))
                    delay = min(delay * 2, 30)
                else:
                    logger.error(f"[VectorStoreAgent] ❌ Non-429 search error: {type(e).__name__}: {e}")
                    return []
        else:
            logger.error("[VectorStoreAgent] ❌ Max retries exceeded while embedding query.")
            return []

        query_vector = response.data[0].embedding

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        output = []
        preview_lines = []
        for idx, hit in enumerate(results):
            text_preview = hit.payload.get("text", "")[:120].replace("\n", " ")
            metadata = hit.payload
            output.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": metadata
            })
            preview_lines.append(
                f"  {idx+1}. score={hit.score:.4f} | preview='{text_preview}...' | meta={ {k:v for k,v in metadata.items() if k!='text'} }"
            )

        if preview_lines:
            logger.info("[VectorStoreAgent] 📊 Retrieved top chunks:\n" + "\n".join(preview_lines))
        else:
            logger.warning("[VectorStoreAgent] ⚠️ No results found in search.")

        return output



    def delete_collection(self):
        logger.info(f"[VectorStoreAgent] Dropping collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        logger.info("[VectorStoreAgent] 🗑️ Collection deleted")


