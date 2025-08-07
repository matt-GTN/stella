import os
import json
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class ResearchPDFHandler:
    """Handles PDF research document processing and querying for Stella"""
    
    def __init__(self, pdf_path: str, persist_directory: str = "./chroma_research_db"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.setup_vectorstore()
    
    def setup_vectorstore(self):
        """Initialize or load the vector store with the research PDF"""
        try:
            if os.path.exists(self.persist_directory):
                # Load existing vectorstore
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings
                )
            else:
                # Create new vectorstore
                self._create_new_vectorstore()
        except Exception as e:
            raise Exception(f"Error setting up vector store: {e}")
    
    def _create_new_vectorstore(self):
        """Create a new vector store from the PDF"""
        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()

            if not documents:
                raise ValueError("PDF loaded but contains no pages")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " "],
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)

            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "source": "research_document",
                    "page_number": chunk.metadata.get("page", 0),
                    "total_chunks": len(chunks)
                })

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )

        except Exception as e:
            raise Exception(f"Error creating vector store from PDF: {e}")
    
    def search_research(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the research document for relevant information"""
        try:
            if not self.vectorstore:
                raise ValueError("Vector store not initialized")

            relevant_docs = self.vectorstore.similarity_search_with_score(query, k=k)

            results = []
            for doc, score in relevant_docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                    "page": doc.metadata.get("page", "Unknown")
                })

            return results

        except Exception as e:
            raise Exception(f"Error searching research: {e}")


# Global handler
RESEARCH_PDF_PATH = "reports/report.pdf"
research_handler = None

def initialize_research_handler():
    """Initialize the research handler - call this when the app starts"""
    global research_handler
    try:
        if os.path.exists(RESEARCH_PDF_PATH):
            research_handler = ResearchPDFHandler(RESEARCH_PDF_PATH)
        else:
            raise FileNotFoundError(f"Research PDF not found at path: {RESEARCH_PDF_PATH}")
    except Exception as e:
        raise Exception(f"Failed to initialize research handler: {e}")

def query_research_document(query: str, max_results: int = 5) -> str:
    """
    Query the research document and return formatted results
    This function will be called by the LangGraph tool
    """
    global research_handler

    if not research_handler:
        return "Research document not available. Please ensure the PDF is properly loaded."

    try:
        results = research_handler.search_research(query, k=max_results)

        if not results:
            return f"No relevant information found in the research document for: '{query}'"

        response_parts = [
            f"Based on our research document, here's what I found about '{query}':\n"
        ]

        for i, result in enumerate(results[:3], 1):
            page_info = f"(Page {result['page']})" if result['page'] != "Unknown" else ""
            response_parts.append(
                f"**{i}. Research Finding {page_info}:**\n"
                f"{result['content']}\n"
            )

        response_parts.append(
            f"\n*This information comes from our internal research document "
            f"(showing {min(3, len(results))} most relevant findings).*"
        )

        return "\n".join(response_parts)

    except Exception as e:
        return f"Error accessing research document: {str(e)}"
