"""Excel RAG core: text chunking, SentenceTransformers embeddings, FAISS and LLM."""
import os
from functools import lru_cache
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

from rag_mcp.analysis_engine import analyze_workbook


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


@lru_cache(maxsize=1)
def _embedding_model(model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class ExcelRAG:
    """Build one combined FAISS index across all detected workbook tables/sheets."""

    def __init__(self):
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.top_k = _positive_int_env("RAG_TOP_K", 8)
        self.max_chunk_chars = _positive_int_env("MAX_CHUNK_CHARS", 2500)
        self.chunk_overlap = _positive_int_env("CHUNK_OVERLAP", 250)
        self.embedding_batch_size = _positive_int_env("EMBEDDING_BATCH_SIZE", 32)
        if self.chunk_overlap >= self.max_chunk_chars:
            raise ValueError("CHUNK_OVERLAP must be smaller than MAX_CHUNK_CHARS.")

        self.embedder = _embedding_model(self.embedding_model_name)
        self.chunks: List[Dict] = []
        self.tables: List[Dict] = []
        self.index = None
        self.dimension = None

        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL")
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-configured")
        self.has_api_key = bool(api_key)

    @staticmethod
    def _cell_to_text(value) -> str:
        if pd.isna(value):
            return ""
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value).strip()

    def _split_text(self, text: str, max_chars: int = None, overlap: int = None) -> List[str]:
        """Split text into max-sized pieces with a configurable character overlap."""
        if not text:
            return []
        max_chars = max_chars or self.max_chunk_chars
        overlap = self.chunk_overlap if overlap is None else overlap
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + max_chars, text_length)
            if end < text_length:
                # Prefer a row boundary close to the limit, without making tiny chunks.
                boundary = text.rfind("\n", start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= text_length:
                break
            next_start = max(start + 1, end - overlap)
            start = next_start
        return chunks

    def _dataframe_to_chunks(
        self, df: pd.DataFrame, sheet_name: str, table_name: str,
        orientation: str, workbook_name: str = "uploaded workbook",
    ) -> List[Dict]:
        df = df.dropna(how="all").copy()
        if df.empty:
            return []
        df.columns = [str(col).strip() or f"Column_{i + 1}" for i, col in enumerate(df.columns)]
        columns = list(df.columns)
        row_texts = []
        for row_number, (_, row) in enumerate(df.iterrows(), start=1):
            parts = []
            for column in columns:
                value = self._cell_to_text(row[column])
                if value:
                    parts.append(f"{column}: {value}")
            if parts:
                row_texts.append(f"Record {row_number}: " + " | ".join(parts))

        if not row_texts:
            return []
        # Keep duplicate-header warnings in every chunk so retrieval does not
        # lose the meaning of generated names such as Revenue__2.
        table_metadata = next((item.get("metadata", {}) for item in self.tables
                               if item.get("metadata", {}).get("sheet") == sheet_name
                               and item.get("metadata", {}).get("table") == table_name), {})
        duplicate_headers = table_metadata.get("duplicate_headers", [])
        duplicate_note = ""
        if duplicate_headers:
            descriptions = []
            for group in duplicate_headers:
                labels = [f"column {entry['position']} ({entry['original']})"
                          for entry in group.get("occurrences", [])]
                descriptions.append(f"'{group['header']}' appears in {', '.join(labels)}")
            duplicate_note = "Duplicate original headers detected: " + "; ".join(descriptions) + ". Internal column names were made unique using __2, __3 suffixes. Treat suffixed columns as separate source columns; do not merge them unless the user confirms they mean the same thing.\n"
        header = (
            f"Workbook: {workbook_name}\nSheet: {sheet_name}\nTable: {table_name}\n"
            f"Layout: {orientation}\nColumns: {', '.join(columns)}\n{duplicate_note}"
        )
        # Keep the table identity in every chunk, even when the body is split.
        body = "\n".join(row_texts)
        prefix = header
        if len(prefix) > self.max_chunk_chars // 2:
            prefix = f"Workbook: {workbook_name} | Sheet: {sheet_name} | Table: {table_name} | Layout: {orientation}\n"
        if len(prefix) >= self.max_chunk_chars:
            prefix = f"Sheet: {sheet_name} | Table: {table_name}\n"
        body_limit = max(1, self.max_chunk_chars - len(prefix))

        # Split body with overlap, then prepend the source metadata.
        body_chunks = self._split_text(
            body,
            max_chars=body_limit,
            # Avoid an overlap so large that the splitter advances only a few characters.
            overlap=min(self.chunk_overlap, max(0, body_limit // 2)),
        )

        output = []
        for chunk_number, body_chunk in enumerate(body_chunks, start=1):
            output.append({
                "text": prefix + body_chunk,
                "sheet": sheet_name,
                "table": table_name,
                "orientation": orientation,
                "rows": body_chunk.count("\n") + 1,
                "chunk_number": chunk_number,
                "workbook": workbook_name,
                "duplicate_headers": duplicate_headers,
            })
        return output

    def build(self, tables: List[Dict], sheet_names: List[str] = None,
              workbook_name: str = "uploaded workbook") -> Dict:
        self.tables = tables  # available while chunk headers are being constructed
        all_chunks = []
        total_rows = 0
        all_columns = set()
        duplicate_header_warnings = []
        detected_sheets = set(sheet_names or [])

        for item in tables:
            df = item["dataframe"]
            metadata = item["metadata"]
            total_rows += len(df)
            detected_sheets.add(metadata["sheet"])
            all_columns.update(str(column) for column in df.columns)
            if metadata.get("duplicate_headers"):
                duplicate_header_warnings.append({
                    "sheet": metadata.get("sheet"),
                    "table": metadata.get("table"),
                    "duplicates": metadata["duplicate_headers"],
                })
            all_chunks.extend(self._dataframe_to_chunks(
                df,
                sheet_name=metadata["sheet"],
                table_name=metadata["table"],
                orientation=metadata["orientation"],
                workbook_name=workbook_name,
            ))

        self.chunks = all_chunks
        # Keep the full ingested DataFrames for exact Pandas calculations.
        self.tables = tables
        self.index = None
        self.dimension = None
        if all_chunks:
            texts = [chunk["text"] for chunk in all_chunks]
            vectors = self.embedder.encode(
                texts,
                batch_size=self.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            self.dimension = int(vectors.shape[1])
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(vectors)

        tables_by_sheet = {}
        for item in tables:
            name = item["metadata"]["sheet"]
            tables_by_sheet[name] = tables_by_sheet.get(name, 0) + 1

        return {
            "sheets": len(detected_sheets),
            "sheet_names": sorted(detected_sheets),
            "tables": len(tables),
            "tables_by_sheet": tables_by_sheet,
            "rows": total_rows,
            "chunks": len(all_chunks),
            "columns": sorted(all_columns),
            "duplicate_header_warnings": duplicate_header_warnings,
            "embedding_model": self.embedding_model_name,
            "top_k": self.top_k,
            "max_chunk_chars": self.max_chunk_chars,
            "chunk_overlap": self.chunk_overlap,
            "embedding_batch_size": self.embedding_batch_size,
        }

    def retrieve(self, question: str, top_k: int = None) -> List[Dict]:
        if self.index is None or not self.chunks:
            return []
        query_vector = self.embedder.encode(
            [question],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        k = min(top_k or self.top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = self.chunks[int(idx)].copy()
                item["score"] = float(score)
                results.append(item)
        return results

    def answer(self, question: str) -> Dict:
        """Hybrid answer: exact Pandas analysis + semantic FAISS retrieval + LLM."""
        retrieved = self.retrieve(question)
        analysis = analyze_workbook(self.tables, question)
        analysis_text = (analysis.get("text") or "")[:12000]

        context = "\n\n".join(
            f"[Source {i + 1} | Sheet: {item['sheet']} | Table: {item['table']} | "
            f"Similarity: {item['score']:.3f}]\n{item['text']}"
            for i, item in enumerate(retrieved)
        )

        if not retrieved and not analysis_text:
            return {
                "answer": "I could not find relevant information in the uploaded workbook.",
                "sources": [],
                "analysis_type": "no_results",
            }

        if not self.has_api_key:
            if analysis.get("handled"):
                answer = (
                    "Pandas calculated this result using the full ingested workbook data. "
                    "LLM_API_KEY is not configured, so this is the deterministic calculation "
                    "without additional natural-language synthesis.\n\n" + analysis_text
                )
                analysis_type = "pandas_numerical"
            else:
                answer = (
                    "Retrieval/profile information is available, but LLM-generated synthesis "
                    "is not enabled. Add LLM_API_KEY to your .env file.\n\n"
                    + analysis_text + ("\n\nRETRIEVED CONTEXT:\n" + context if context else "")
                )
                analysis_type = "rag_retrieval"
        else:
            system_prompt = (
                "You are a careful Excel data analyst using a hybrid Pandas + RAG system. "
                "PANDAS ANALYSIS is computed directly from the full ingested workbook tables "
                "and must be treated as the source of truth for numerical calculations. Do not "
                "recalculate or contradict those figures. Retrieved RAG context is selected "
                "evidence for semantic questions and may represent only a subset of rows. "
                "Use both sources when useful, combine evidence across worksheets, and name "
                "the source sheet(s). Excel headers may be duplicated. When a duplicate-header "
                "warning appears, names with suffixes such as __2 represent distinct original "
                "columns at different positions. Do not silently combine them or assume they "
                "have identical meanings. If the question says only the unsuffixed name and "
                "the intended column is unclear, explain the ambiguity and ask which one is meant. "
                "For numerical questions, report the Pandas result and "
                "explain it clearly. For descriptive or semantic questions, summarize patterns "
                "from the workbook profile and retrieved context. If evidence is insufficient "
                "or a column match is ambiguous, say so rather than inventing information. "
                "Do not claim a causal explanation unless the workbook supports it."
            )
            user_context = (
                f"FULL-WORKBOOK PANDAS ANALYSIS:\n{analysis_text or 'No calculation/profile available.'}\n\n"
                f"RETRIEVED RAG CONTEXT (up to {self.top_k} chunks):\n"
                f"{context or 'No relevant chunks retrieved.'}\n\n"
                f"USER QUESTION: {question}"
            )
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context},
                ],
            )
            answer = response.choices[0].message.content or "The model returned an empty answer."
            analysis_type = "pandas_numerical" if analysis.get("handled") else "rag_and_descriptive"

        if any(item.get("duplicate_headers") for item in self.tables):
            duplicate_summary = []
            for table_item in self.tables:
                meta = table_item.get("metadata", {})
                for group in meta.get("duplicate_headers", []):
                    duplicate_summary.append(
                        f"{meta.get('sheet')} / {meta.get('table')}: '{group.get('header')}'"
                    )
            if duplicate_summary:
                # Surface a visible note even when the LLM does not mention it.
                answer = ("**Data quality note — duplicate headers detected:** "
                          + "; ".join(duplicate_summary) + ". Columns were assigned unique internal names (for example, __2) and treated separately.\n\n"
                          + answer)

        sources = [{
            "sheet": item["sheet"],
            "table": item["table"],
            "orientation": item["orientation"],
            "rows": item["rows"],
            "chunk_number": item["chunk_number"],
            "score": round(item["score"], 3),
            "preview": item["text"][:650],
        } for item in retrieved]
        return {
            "answer": answer,
            "sources": sources,
            "analysis_type": analysis_type,
            "pandas_analysis_used": bool(analysis.get("handled")),
        }
