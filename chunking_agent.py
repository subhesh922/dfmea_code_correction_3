# server/agents/chunking_agent.py

import uuid
import tiktoken
import warnings
from tqdm import tqdm
from typing import List, Dict
import tiktoken

class ChunkingAgent:
    def __init__(self, max_tokens=1000, overlap=50, model_name="text-embedding-3-small"):
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.global_stats = {"total_chunks": 0, "total_tokens": 0, "sources": {}}
        warnings.filterwarnings("ignore", category=UserWarning)

    def run(self, prd_data: List[Dict], kb_data: List[Dict], fi_data: List[Dict]) -> List[Dict]:
        print("[DEBUG] ChunkingAgent.run executed, chunks created with metadata.")
        
        print(f"[ChunkingAgent] Chunking PRDs...")
        prd_chunks = self._create_chunks(prd_data, source="prds")

        print(f"[ChunkingAgent] Chunking knowledge bank...")
        kb_chunks = self._create_chunks(kb_data, source="knowledge_bank")

        print(f"[ChunkingAgent] Chunking field-reported issues...")
        fi_chunks = self._create_chunks(fi_data, source="field_issues")

        all_chunks = prd_chunks + kb_chunks + fi_chunks
        print(f"[ChunkingAgent] Merged into {len(all_chunks)} smart chunks before token slicing.")

        sliced_chunks = self._token_slice_chunks(all_chunks)
        return sliced_chunks

    def _create_chunks(self, data: List[Dict], source: str) -> List[Dict]:
        chunks = []
        for row in data:
            text = self._format_row_as_text(row)
            if text.strip():
                chunks.append({
                    "text": text,
                    "metadata": {
                        "uuid": str(uuid.uuid4()),
                        "source": source
                    }
                })
        return chunks

    def _format_row_as_text(self, row: Dict) -> str:
        return " | ".join(
            f"{k.strip()}: {str(v).strip()}"
            for k, v in row.items()
            if v and str(v).strip()
        )

    def _token_slice_chunks(self, chunks: List[Dict]) -> List[Dict]:
        sliced_chunks = []
        local_stats = {}
        for chunk in tqdm(chunks, desc="[ChunkingAgent] Token slicing"):
            source = chunk["metadata"].get("source", "unknown")
            tokens = self.encoder.encode(chunk["text"])

            if source not in local_stats:
                local_stats[source] = {"tokens": 0, "chunks": 0}
            local_stats[source]["tokens"] += len(tokens)

            if len(tokens) <= self.max_tokens:
                sliced_chunks.append(chunk)
                local_stats[source]["chunks"] += 1
                continue

            start = 0
            while start < len(tokens):
                end = min(start + self.max_tokens, len(tokens))
                token_slice = tokens[start:end]
                text_slice = self.encoder.decode(token_slice)

                sliced_chunks.append({
                    "text": text_slice,
                    "metadata": chunk["metadata"]
                })
                local_stats[source]["chunks"] += 1

                if end == len(tokens):
                    break
                start += self.max_tokens - self.overlap

        # Update global stats
        for src, stats in local_stats.items():
            if src not in self.global_stats["sources"]:
                self.global_stats["sources"][src] = {"tokens": 0, "chunks": 0}
            self.global_stats["sources"][src]["tokens"] += stats["tokens"]
            self.global_stats["sources"][src]["chunks"] += stats["chunks"]

        self.global_stats["total_chunks"] += sum(v["chunks"] for v in local_stats.values())
        self.global_stats["total_tokens"] += sum(v["tokens"] for v in local_stats.values())

        return sliced_chunks

    def print_summary(self):
        print("\n[ChunkingAgent] === Chunking Summary ===")
        print(f"Total Chunks: {self.global_stats['total_chunks']}")
        print(f"Total Tokens: {self.global_stats['total_tokens']}")
        for src, stats in self.global_stats["sources"].items():
            print(f"  └── {src}: {stats['chunks']} chunks, {stats['tokens']} tokens")


