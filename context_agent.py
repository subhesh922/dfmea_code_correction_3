import re
import json
import asyncio
from typing import List, Dict, Optional
from .vectorstore_agent import VectorStoreAgent


class ContextAgent:
    def __init__(self, llm_client, batch_size: int = 10, collection_name: str = "dfmea_collection"):
        self.llm_client = llm_client
        self.batch_size = batch_size
        # 🔑 always bind to a specific collection name
        self.vectorstore = VectorStoreAgent(collection_name=collection_name)


    def _build_prompt(
    self,
    products: List[str],
    subproducts: List[str],
    focus: Optional[str],
    prd_chunks: List[str],
    kb_chunks: List[str],
    field_chunks: List[str],
) -> str:
        """Zebra-specific DFMEA prompt with strict product/subproduct mapping."""

        return f"""
    You are a senior DFMEA analyst at Zebra Technologies.

    Task:
    Generate hardware-focused DFMEA entries strictly in JSON format. 
    Do not include explanations, commentary, or markdown formatting. 
    The output must be valid JSON that can be parsed directly.

    Context:
    - Products: {", ".join(products)}
    - Subproducts: {", ".join(subproducts)}
    - PRD Evidence (sample):{ "\n".join(prd_chunks[:10]) }
    - Knowledge Base Evidence (sample):{ "\n".join(kb_chunks[:10]) }
    - Field Issue Evidence (sample):{ "\n".join(field_chunks[:10]) }
    
    

    Instructions:
    - Create **one DFMEA entry for every unique issue, failure mode, or risk** found in the chunks.
    - Each entry **must be tied to exactly ONE product from the provided product list**.
    - Each entry **must also be tied to exactly ONE subproduct from the provided subproduct list**.
    - Do **NOT** use generic placeholders like "All". Always pick the most relevant product and subproduct.
    - If multiple products or subproducts are relevant, duplicate the entry and assign one product/subproduct per entry.
    - At least **one entry per batch must explicitly leverage field issue evidence**.

    Required JSON Output:
    {{
    "entries": [
        {{
        "Product": "one specific product from [{', '.join(products)}]",
        "Subproducts": "one specific subproduct from [{', '.join(subproducts)}]",
        "Function": "Component purpose from PRD",
        "Potential Failure Mode": "Hardware failure description",
        "Potential Effects": ["Impact on device operation"],
        "Potential Causes": ["Root causes (design, manufacturing, usage)"],
        "Severity": 1-10,
        "Occurrence": 1-10,
        "Detection": 1-10,
        "RPN": "Must equal Severity × Occurrence × Detection (integer only)",
        "Controls Prevention": ["Prevention actions"],
        "Controls Detection": ["Detection/QA methods"],
        "linked_to_kb": true
        }}
    ]
    }}

    Zebra-Specific Rules:
    1. Base entries on the provided evidence; where gaps exist, infer intelligently from field_issues.
    2. Return **pure valid JSON** only (no markdown, no commentary).
    3. Generate 1–3 DFMEA entries per batch.
    4. Ensure numeric fields are integers, never leave arrays empty, focus exclusively on hardware issues, and avoid redundant or generic statements.
    5. Apply Zebra severity scale:
    - 9-10: Safety/Legal impact
    - 7-8: Device inoperable
    - 4-6: Reduced performance
    - 1-3: Cosmetic
    
    """ + (f"\nZebra Engineering Focus: {focus}" if focus else "")



    def _parse_llm_response(self, raw_response: str) -> List[Dict]:
        """Cleans and parses LLM JSON output."""
        cleaned = re.sub(r"^```(json)?", "", raw_response.strip(), flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

        try:
            parsed = json.loads(cleaned)
            return parsed.get("entries", []) if isinstance(parsed, dict) else []
        except json.JSONDecodeError:
            print("[ContextAgent] JSON decode failed after cleanup.")
            return []

    def _call_azure_openai(self, prompt: str) -> str:
        """Synchronous wrapper to call Azure OpenAI."""
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",  # update if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    def _batch(self, chunks: List[str], size: int) -> List[List[str]]:
        """Split list into batches."""
        return [chunks[i : i + size] for i in range(0, len(chunks), size)]

    def _process_batch(
        self,
        batch_chunks: List[str],
        products: List[str],
        subproducts: List[str],
        focus: Optional[str],
        prd_chunks: List[str],
        kb_chunks: List[str],
        field_chunks: List[str],
    ) -> List[Dict]:
        """Process one batch of chunks through LLM."""
        user_msg = "Here are relevant data chunks:\n\n" + "\n\n".join(batch_chunks)
        prompt = self._build_prompt(products, subproducts, focus, prd_chunks, kb_chunks, field_chunks) + "\n\n" + user_msg

        raw_response = self._call_azure_openai(prompt)
        parsed = self._parse_llm_response(raw_response)

        return parsed

    # def run(
    #     self,
    #     query: str,
    #     products: List[str],
    #     subproducts: List[str],
    #     focus: Optional[str] = None,
    #     top_k: int = 200,
    # ) -> List[Dict]:
    #     """Sync entrypoint: search Qdrant → build prompt → generate DFMEA JSON."""

    #     print(f"[ContextAgent] Searching Qdrant for: {query}")
    #     matches = self.vectorstore.search(query, top_k=top_k)
    #     chunks = [m["text"] for m in matches]

    #     print(f"[ContextAgent] Retrieved {len(chunks)} chunks from Qdrant.")

    #     # For prompt stats
    #     prd_chunks = [c for c in chunks if "prd" in c.lower()]
    #     kb_chunks = [c for c in chunks if "kb" in c.lower()]
    #     field_chunks = [c for c in chunks if "field" in c.lower()]

    #     results = []
    #     for batch in self._batch(chunks, self.batch_size):
    #         results.extend(
    #             self._process_batch(batch, products, subproducts, focus, prd_chunks, kb_chunks, field_chunks)
    #         )

    #     print(f"\n[ContextAgent] Parsed {len(results)} DFMEA entries.\n")
    #     return results
    async def run(
    self,
    query: str,
    products: List[str],
    subproducts: List[str],
    focus: Optional[str] = None,
    top_k: int = 200,
    chunk_cap: int = 200,   # 👈 max chunks per product+subproduct
    max_concurrent: int = 5 # 👈 tune this to control parallelism
) -> List[Dict]:
        """Parallel (semaphore-limited) search Qdrant for each product+subproduct pair → DFMEA JSON."""

        results = []
        total_pairs = len(products) * len(subproducts)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_pair(idx: int, product: str, subproduct: str):
            async with semaphore:
                print(f"\n[ContextAgent] 🔎 Processing {idx}/{total_pairs} → Product: {product}, Subproduct: {subproduct}")

                # 🔹 Run search (blocking → thread executor)
                matches = await asyncio.to_thread(self.vectorstore.search, query, top_k=top_k)
                chunks = [m["text"] for m in matches]

                # Cap chunks
                if len(chunks) > chunk_cap:
                    chunks = chunks[:chunk_cap]

                print(f"[ContextAgent] Retrieved {len(chunks)} capped chunks for {product} - {subproduct}.")

                # ✅ Split by reliable Qdrant metadata
                prd_chunks   = [m["text"] for m in matches if m.get("payload", {}).get("source") == "prds"]
                kb_chunks    = [m["text"] for m in matches if m.get("payload", {}).get("source") == "knowledge_base"]
                field_chunks = [m["text"] for m in matches if m.get("payload", {}).get("source") == "field_issues"]

                # ✅ Fallback: if any set is empty, use general chunks
                if not prd_chunks:
                    print(f"[ContextAgent] ⚠️ No PRD chunks found → falling back to general chunks")
                    prd_chunks = chunks
                if not kb_chunks:
                    print(f"[ContextAgent] ⚠️ No KB chunks found → falling back to general chunks")
                    kb_chunks = chunks
                if not field_chunks:
                    print(f"[ContextAgent] ⚠️ No FIELD chunks found → falling back to general chunks")
                    field_chunks = chunks

                pair_results = []
                for batch in self._batch(chunks, self.batch_size):
                    batch_results = await asyncio.to_thread(
                        self._process_batch,
                        batch,
                        [product],
                        [subproduct],
                        focus,
                        prd_chunks,
                        kb_chunks,
                        field_chunks,
                    )
                    pair_results.extend(batch_results)

                print(f"[ContextAgent] ✅ Completed {product} - {subproduct}, results: {len(pair_results)}")
                return pair_results

        # 🔹 Launch all pairs concurrently (with semaphore limit)
        tasks = [
            process_pair(idx, product, subproduct)
            for idx, (product, subproduct) in enumerate(
                [(p, s) for p in products for s in subproducts], start=1
            )
        ]

        all_results = await asyncio.gather(*tasks)
        for r in all_results:
            results.extend(r)

        print(f"\n[ContextAgent] 🎯 Finished. Parsed {len(results)} DFMEA entries across {total_pairs} pairs.\n")
        return results


#     async def run(
#     self,
#     query: str,
#     products: List[str],
#     subproducts: List[str],
#     focus: Optional[str] = None,
#     top_k: int = 200,
#     chunk_cap: int = 200,   # 👈 max chunks per product+subproduct
#     max_concurrent: int = 5 # 👈 tune this to control parallelism
# ) -> List[Dict]:
#         """Parallel (semaphore-limited) search Qdrant for each product+subproduct pair → DFMEA JSON."""

#         results = []
#         total_pairs = len(products) * len(subproducts)
#         semaphore = asyncio.Semaphore(max_concurrent)

#         async def process_pair(idx: int, product: str, subproduct: str):
#             async with semaphore:
#                 print(f"\n[ContextAgent] 🔎 Processing {idx}/{total_pairs} → Product: {product}, Subproduct: {subproduct}")

#                 # 🔹 Run search (blocking → thread executor)
#                 matches = await asyncio.to_thread(self.vectorstore.search, query, top_k=top_k)
#                 chunks = [m["text"] for m in matches]

#                 # Cap chunks
#                 if len(chunks) > chunk_cap:
#                     chunks = chunks[:chunk_cap]

#                 print(f"[ContextAgent] Retrieved {len(chunks)} capped chunks for {product} - {subproduct}.")

#                 # Split chunks by type
#                 prd_chunks = [c for c in chunks if "prd" in c.lower()]
#                 kb_chunks = [c for c in chunks if "kb" in c.lower()]
#                 field_chunks = [c for c in chunks if "field" in c.lower()]

#                 pair_results = []
#                 for batch in self._batch(chunks, self.batch_size):
#                     batch_results = await asyncio.to_thread(
#                         self._process_batch,
#                         batch,
#                         [product],
#                         [subproduct],
#                         focus,
#                         prd_chunks,
#                         kb_chunks,
#                         field_chunks,
#                     )
#                     pair_results.extend(batch_results)

#                 print(f"[ContextAgent] ✅ Completed {product} - {subproduct}, results: {len(pair_results)}")
#                 return pair_results

#         # 🔹 Launch all pairs concurrently (with semaphore limit)
#         tasks = [
#             process_pair(idx, product, subproduct)
#             for idx, (product, subproduct) in enumerate(
#                 [(p, s) for p in products for s in subproducts], start=1
#             )
#         ]

#         all_results = await asyncio.gather(*tasks)
#         for r in all_results:
#             results.extend(r)

#         print(f"\n[ContextAgent] 🎯 Finished. Parsed {len(results)} DFMEA entries across {total_pairs} pairs.\n")
#         return results





