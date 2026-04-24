"""Agentic RAG tools: index and search documents for knowledge retrieval."""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config import get_settings
from ..observability import emit

# [IMPROVE-69] See tools/file_ops.py for the WORKSPACE_ROOT rationale —
# these two modules share the workspace sandbox contract and migrate
# together.
WORKSPACE_ROOT = Path(get_settings().local_ai_workspace).resolve()


class IndexDocumentInput(BaseModel):
    path: str = Field(..., description="Path to a text file to index, relative to workspace")
    collection: str = Field("default", description="Collection name to index into")


class SearchDocumentsInput(BaseModel):
    query: str = Field(..., description="Natural language search query")
    collection: str = Field("default", description="Collection to search in")
    max_results: int = Field(5, description="Maximum results to return")


class ListCollectionsInput(BaseModel):
    pass


# Lazy singleton to avoid importing chromadb at module load time
_vector_memory = None


def _get_vector_memory():
    global _vector_memory
    if _vector_memory is None:
        from local_ai_platform.memory import VectorMemory
        _vector_memory = VectorMemory(collection_name="rag_documents", persist_dir="./data/vectorstore")
    return _vector_memory


def _safe_path(user_path: str) -> Path:
    """Resolve a user-provided path and verify it's inside the workspace.

    Mirrors tools/file_ops._safe_path — see that file for the rationale
    behind relative_to() vs the old startswith() check (sibling-prefix
    escape: WORKSPACE_ROOT=/home/a/workspace would accept a resolved path
    /home/a/workspace_other/secret.txt). Both helpers emit the same
    "file_ops.path_rejected" event so observability has a single signal
    for sandbox-escape attempts regardless of which tool was invoked.
    """
    resolved = (WORKSPACE_ROOT / user_path).resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        emit(
            "tool",
            "file_ops.path_rejected",
            status="error",
            error_code="PathOutsideWorkspace",
            context={"user_path": user_path[:200]},
        )
        raise ValueError(f"Path '{user_path}' is outside the workspace directory")
    return resolved


def index_document(path: str, collection: str = "default") -> str:
    """Index a text file for later semantic search retrieval."""
    try:
        fp = _safe_path(path)
        if not fp.exists():
            return f"File not found: {path}"
        if not fp.is_file():
            return f"Not a file: {path}"

        content = fp.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return f"File is empty: {path}"

        vm = _get_vector_memory()
        if vm._get_collection() is None:
            return "Vector store not available. Install chromadb: pip install chromadb"

        # Split content into chunks for better retrieval
        chunk_size = 500
        chunks = []
        words = content.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        from local_ai_platform.providers import ChatMessage
        messages = [ChatMessage(role="document", content=chunk) for chunk in chunks]
        vm.store(
            conversation_id=f"doc:{collection}:{fp.name}",
            messages=messages,
            metadata={"source": str(fp.name), "collection": collection},
        )

        return f"Indexed {len(chunks)} chunks from {path} into collection '{collection}'"

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Indexing failed: {e}"


def search_documents(query: str, collection: str = "default", max_results: int = 5) -> str:
    """Search indexed documents using semantic similarity."""
    try:
        vm = _get_vector_memory()
        if vm._get_collection() is None:
            return "Vector store not available. Install chromadb: pip install chromadb"

        results = vm.search(query, n_results=max_results)
        if not results:
            return f"No relevant documents found for: {query}"

        lines = []
        for r in results:
            source = r["metadata"].get("source", "unknown")
            relevance = r.get("relevance", 0)
            content = r["content"][:300]
            lines.append(f"[{source}] (relevance: {relevance:.2f})\n{content}")

        return f"Found {len(results)} results:\n\n" + "\n\n---\n\n".join(lines)

    except Exception as e:
        return f"Search failed: {e}"


def get_rag_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=index_document,
            name="index_document",
            description="Index a text file from the workspace for semantic search. Use this to build a knowledge base from documents.",
            args_schema=IndexDocumentInput,
        ),
        StructuredTool.from_function(
            func=search_documents,
            name="search_documents",
            description="Search indexed documents using natural language. Returns relevant passages with source info.",
            args_schema=SearchDocumentsInput,
        ),
    ]
