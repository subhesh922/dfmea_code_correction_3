
from fastapi import FastAPI, UploadFile, File
from typing import List,Optional
from pathlib import Path
from .agents.chunking_agent import ChunkingAgent
from .agents.embedding_agent import EmbeddingAgent
from .agents.vectorstore_agent import VectorStoreAgent
from .agents.context_agent import ContextAgent
from .utils.azure_openai_client import client
from .utils.file_parser import parse_file
from .agents.embedding_agent import EmbeddingAgent
from .utils.logger import logger
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tempfile
from fastapi.responses import FileResponse
import logging
import time 
import sys 
import os 

for noisy in ["httpx", "openai"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chunker = ChunkingAgent()
embedder = EmbeddingAgent()


@app.post("/dfmea/generate")
async def generate_dfmea(
    products: List[str] = Form(...),
    subproducts: List[str] = Form(...),
    focus: Optional[str] = Form(None),
    prds: List[UploadFile] = File(None),
    knowledge_base: List[UploadFile] = File(None),
    field_issues: List[UploadFile] = File(None)
):
    try:
        # 🔹 Log frontend inputs
        logger.info("📥 [Frontend Input] Products: %s", products)
        logger.info("📥 [Frontend Input] Subproducts: %s", subproducts)
        logger.info("📥 [Frontend Input] Focus: %s", focus if focus else "None")

        # Step 1: Buckets for parsed data
        prd_data, kb_data, fi_data = [], [], []

        # Step 2: File processor
        async def process_files(files, bucket: list, label: str):
            if not files:
                return
            for f in files:
                tmp_path = Path(f.filename)
                with open(tmp_path, "wb") as buffer:
                    buffer.write(await f.read())

                parsed_data = parse_file(tmp_path)
                logger.info(f"[Parser] Parsed {f.filename} ({len(parsed_data)} chars)")
                bucket.extend(parsed_data)

        # Step 3: Parse all files into buckets
        await process_files(prds, prd_data, "prds")
        await process_files(knowledge_base, kb_data, "knowledge_base")
        await process_files(field_issues, fi_data, "field_issues")

        # Step 4: Chunk data
        all_chunks = chunker.run(prd_data, kb_data, fi_data)
        logger.info(f"[Chunker] ✅ Created {len(all_chunks)} total chunks via run()")

        # Step 5: Count source-wise chunks
        prd_chunks = sum(1 for c in all_chunks if c["metadata"]["source"] == "prds")
        kb_chunks = sum(1 for c in all_chunks if c["metadata"]["source"] == "knowledge_base")
        fi_chunks = sum(1 for c in all_chunks if c["metadata"]["source"] == "field_issues")
        total_chunks = len(all_chunks)

        # # Step 6: Embedding
        # embedded_chunks = []
        # if all_chunks:
        #     batch_size = 50
        #     total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        #     logger.info(f"[EmbeddingAgent] 🚀 Starting embeddings for {len(all_chunks)} chunks")
        #     for i in range(total_batches):
        #         start = i * batch_size
        #         end = min((i + 1) * batch_size, len(all_chunks))
        #         batch = all_chunks[start:end]

        #         batch_embedded = await embedder.embed_chunks_async(batch)
        #         embedded_chunks.extend(batch_embedded)

        #         # pct = int(((i + 1) / total_batches) * 100)
        #         # sys.stdout.write(f"\r[Embedding Progress] {pct}%\n")
        #         # sys.stdout.flush()
        #         # time.sleep(0.05)

        #     sys.stdout.write("\n")
        #     logger.info(f"[EmbeddingAgent] ✅ Completed embeddings: {len(all_chunks)} total")
        # Step 6: Embedding
        embedded_chunks = []
        if all_chunks:
            batch_size = 50
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size

            logger.info(f"[EmbeddingAgent] 🚀 Starting embeddings for {len(all_chunks)} chunks "
                        f"(batch_size={batch_size}, total_batches={total_batches})")

            # for i in range(total_batches):
            #     start = i * batch_size
            #     end = min((i + 1) * batch_size, len(all_chunks))
            #     batch = all_chunks[start:end]

            #     batch_embedded = await embedder.embed_chunks_async(batch)
            #     embedded_chunks.extend(batch_embedded)

            #     # Log progress every 5% or every 10 batches
            #     if (i + 1) % max(1, total_batches // 20) == 0 or (i + 1) == total_batches:
            #         pct = int(((i + 1) / total_batches) * 100)
            #         logger.info(f"[EmbeddingAgent] 📊 Progress: {pct}% "
            #                     f"({i+1}/{total_batches} batches done)")
            # Do embedding once (no batching here)
            embedded_chunks = await embedder.embed_chunks_async(all_chunks)

            # Now run a dummy loop just for progress logs
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            for i in range(total_batches):
                pct = int(((i + 1) / total_batches) * 100)
                logger.info(f"[EmbeddingAgent] 📊 Progress: {pct}% "
                            f"({i+1}/{total_batches} batches done)")

            # logger.info(f"[EmbeddingAgent] ✅ Completed embeddings: {len(embedded_chunks)} total")


        # Step 7: Insert into Qdrant (fixed collection name)
        collection_name = "dfmea_collection"
        if embedded_chunks:
            vectorstore = VectorStoreAgent(collection_name=collection_name)
            vectorstore.create_collection(vector_dim=len(embedded_chunks[0]["embedding"]))
            vectorstore.add_embeddings(embedded_chunks)
            logger.info(f"[VectorStore] ✅ Inserted {len(embedded_chunks)} vectors into Qdrant")

        # 🔹 Step 8: ContextAgent execution
        dfmea_entries = []   # ensure it always exists
        try:
            context_agent = ContextAgent(
                llm_client=client,
                batch_size=10,
                collection_name=collection_name
            )
            dfmea_entries = await context_agent.run(
                query="DFMEA for Zebra hardware",
                products=products,
                subproducts=subproducts,
                focus=focus,
                top_k=50
            )
        except Exception as ce:
            logger.error(f"[ContextAgent] ❌ Error while running DFMEA: {ce}")

        # ✅ Step 9: Final JSON Response
        return {
            "status": "success",
            "embedding_summary": {
                "total_vectors": total_chunks,
                "prd_vectors": prd_chunks,
                "kb_vectors": kb_chunks,
                "fi_vectors": fi_chunks
            },
            "dfmea_entries": dfmea_entries
        }

    except Exception as e:
        logger.error(f"[DFMEA] ❌ Error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    try:
        file_path = Path(tempfile.gettempdir()) / f"{file_id}.xlsx"

        if not file_path.exists():
            return {"status": "error", "message": "File not found"}

        # 🔹 Serve file
        response = FileResponse(
            path=file_path,
            filename=f"DFMEA_{file_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # 🔹 Delete file after serving
        @response.call_on_close
        def cleanup():
            try:
                os.remove(file_path)
            except Exception:
                pass

        return response
    except Exception as e:
        return {"status": "error", "message": str(e)}






