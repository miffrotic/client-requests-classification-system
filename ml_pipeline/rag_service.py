from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ml_pipeline.utils.data import parse_intent_value


logger = logging.getLogger(__name__)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_CHROMA_DIR: Final[Path] = _PROJECT_ROOT / "chroma_db"
_DEFAULT_COLLECTION: Final[str] = "store_policies"
_EMBEDDING_MODEL: Final[str] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_SYSTEM_INSTRUCTION: Final[str] = (
    "You are a professional customer-support agent. "
    "Answer the user's question ONLY using the provided context from official "
    "company policy documents. "
    "Reply in the SAME language as the user's question "
    "(English question → English answer, Russian question → Russian answer). "
    "If the context does not contain the answer, say exactly: "
    "'I cannot find exact information about your question in our policies — "
    "I am transferring you to a human agent.'\n"
    "Context:\n{context}"
)

_DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"

load_dotenv(_PROJECT_ROOT / ".env")


def _resolve_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        msg = "GOOGLE_API_KEY (или GEMINI_API_KEY) не найден в .env"
        raise RuntimeError(msg)
    return api_key


def _embedding_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class StorePolicyRAG:

    def __init__(
        self,
        *,
        chroma_dir: Path | str | None = None,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._chroma_dir = Path(chroma_dir) if chroma_dir is not None else _DEFAULT_CHROMA_DIR
        self._collection_name = collection_name
        self._vector_store: Chroma | None = None
        self._qa_chain = None

        device = _embedding_device()
        logger.info("StorePolicyRAG: embedding model device=%s", device)

        self._embeddings = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL,
            model_kwargs={"device": device},
        )
        gemini_model = os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
        self._llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=_resolve_google_api_key(),
            temperature=0.1,
        )

    def build_or_load_index(self, pdf_path: str) -> None:
        pdf = Path(pdf_path)
        if not pdf.is_file():
            msg = f"PDF not found: {pdf}"
            raise FileNotFoundError(msg)

        if self._chroma_dir.is_dir() and any(self._chroma_dir.iterdir()):
            logger.info("Loading existing Chroma index from %s", self._chroma_dir)
            self._vector_store = Chroma(
                persist_directory=str(self._chroma_dir),
                embedding_function=self._embeddings,
                collection_name=self._collection_name,
            )
        else:
            logger.info("Building Chroma index from %s", pdf)
            documents = PyPDFLoader(str(pdf)).load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            ).split_documents(documents)

            self._chroma_dir.mkdir(parents=True, exist_ok=True)
            self._vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self._embeddings,
                persist_directory=str(self._chroma_dir),
                collection_name=self._collection_name,
            )

        retriever = self._vector_store.as_retriever(search_kwargs={"k": 3})
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_INSTRUCTION),
                ("human", "{input}"),
            ],
        )
        document_chain = create_stuff_documents_chain(self._llm, prompt)
        self._qa_chain = create_retrieval_chain(retriever, document_chain)

    def ask(self, query: str) -> str:
        if self._qa_chain is None:
            msg = "Index is not ready. Call build_or_load_index() first."
            raise RuntimeError(msg)

        result = self._qa_chain.invoke({"input": query})
        answer = result.get("answer", "")
        return str(answer).strip()



RAG_POLICY_INTENTS: Final[frozenset[str]] = frozenset(
    {
        "get_refund",
        "track_refund",
        "check_refund_policy",
        "check_cancellation_fee",
    },
)


def parse_classifier_intents(classification: dict | None) -> list[str]:
    if not classification:
        return []
    raw = classification.get("intents")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return parse_intent_value(raw)


def should_route_to_rag(classification: dict | None) -> bool:
    predicted = parse_classifier_intents(classification)
    return bool(RAG_POLICY_INTENTS.intersection(predicted))
