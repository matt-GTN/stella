# agent/src/pdf_research.py
import os
import streamlit as st
from pathlib import Path
import json
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class ResearchPDFHandler:
    """Gère le traitement et l'interrogation de documents de recherche PDF - optimisé pour le contenu financier/science des données"""
    
    def __init__(self, pdf_path: str, persist_directory: str = None):
        self.pdf_path = pdf_path
        # Auto-detect environment and set appropriate path
        if persist_directory is None:
            if os.path.exists("/app"):  # Docker/production environment
                self.persist_directory = "/app/chroma_research_db"
            else:  # Local development environment
                base_dir = Path(__file__).resolve().parents[2]
                self.persist_directory = str(base_dir / "chroma_research_db")
        else:
            self.persist_directory = persist_directory
        self.vectorstore = None
        self.setup_vectorstore()
    
    def setup_vectorstore(self):
        """Initialise ou charge le magasin de vecteurs avec le document de recherche PDF"""
        try:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                # Charge le magasin de vecteurs existant
                embeddings = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-small",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings
                )
                print(f"Magasin de vecteurs existant chargé avec {self.vectorstore._collection.count()} documents")
            else:
                # Crée un nouveau magasin de vecteurs
                self._create_new_vectorstore()
        except Exception as e:
            raise Exception(f"Erreur lors de la configuration du magasin de vecteurs : {e}")
    
    def _create_new_vectorstore(self):
        """Crée un nouveau magasin de vecteurs à partir du PDF"""
        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF introuvable à : {self.pdf_path}")

            print("Chargement du document PDF...")
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()

            if not documents:
                raise ValueError("PDF chargé mais ne contient aucune page")

            print(f"Chargé {len(documents)} pages du PDF")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", "• ", "- ", " "],
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)

            print(f"Divisé en {len(chunks)} morceaux")

            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "source": "rapport_de_recherche_financière",
                    "page_number": chunk.metadata.get("page", 0),
                    "total_chunks": len(chunks),
                    "language": "multilingual",
                    "domain": "financial_data_science"
                })

            print("Création des embeddings avec le modèle multilingue E5...")
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-small",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            print("Construction de la base de données vectorielle...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"Base de données vectorielle créée avec succès avec {len(chunks)} morceaux")

        except Exception as e:
            raise Exception(f"Erreur lors de la création du magasin de vecteurs à partir du PDF : {e}")
    
    def search_research(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recherche des informations pertinentes dans le document de recherche"""
        try:
            if not self.vectorstore:
                raise ValueError("Magasin de vecteurs non initialisé")

            relevant_docs = self.vectorstore.similarity_search_with_score(query, k=k)

            results = []
            for doc, score in relevant_docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                    "page": doc.metadata.get("page", "Inconnu"),
                    "chunk_id": doc.metadata.get("chunk_id", 0)
                })

            return results

        except Exception as e:
            raise Exception(f"Erreur lors de la recherche dans le document de recherche : {e}")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Obtient des statistiques sur le document chargé"""
        if not self.vectorstore:
            return {"error": "Magasin de vecteurs non initialisé"}
        
        try:
            total_chunks = self.vectorstore._collection.count()
            return {
                "total_chunks": total_chunks,
                "pdf_path": self.pdf_path,
                "language": "multilingual",
                "domain": "Financial Data Science",
                "model": "intfloat/multilingual-e5-small"
            }
        except:
            return {"error": "Impossible de récupérer les statistiques du document"}

    @staticmethod
    @st.cache_resource
    def initialize_research_handler():
        """Initialise le gestionnaire de recherche - à appeler au démarrage de l'application"""
        try:
            base_dir = Path(__file__).resolve().parents[2]
            research_pdf_path = base_dir / "reports" / "Rapport de projet - OPA - NOV24-CDS.pdf"
            if os.path.exists(research_pdf_path):
                research_handler = ResearchPDFHandler(str(research_pdf_path))
                print("Gestionnaire de recherche initialisé avec succès")
                return research_handler
            else:
                raise FileNotFoundError(f"PDF de recherche introuvable à : {research_pdf_path}")
        except Exception as e:
            raise Exception(f"Échec de l'initialisation du gestionnaire de recherche : {e}")

# Global variable to store the research handler
_research_handler = None

def query_research_document(query: str, max_results: int = 5) -> str:
    """
    Interroge le document de recherche et renvoie les résultats formatés
    Amélioré pour le contenu de la science des données financières
    """
    global _research_handler
    
    # Lazy initialization - only create when actually needed
    if _research_handler is None:
        try:
            _research_handler = ResearchPDFHandler.initialize_research_handler()
        except Exception as e:
            return f"Erreur lors de l'initialisation du gestionnaire de recherche : {str(e)}"

    if not _research_handler:
        return "Document de recherche non disponible. Veuillez vous assurer que le PDF est correctement chargé."

    try:
        results = _research_handler.search_research(query, k=max_results)

        if not results:
            return f"Aucune information pertinente trouvée dans le document de recherche pour : '{query}'"

        response_parts = [
            f"D'après notre rapport de recherche en analyse financière, voici ce que j'ai trouvé concernant '{query}' :\n"
        ]

        for i, result in enumerate(results[:3], 1):
            page_info = f"(Page {result['page']})" if result['page'] != "Inconnu" else ""
            relevance = f"(Score : {result['relevance_score']:.3f})" if result['relevance_score'] < 1.0 else ""
            
            response_parts.append(
                f"**{i}. Résultat de recherche {page_info} {relevance} :**\n"
                f"{result['content']}\n"
            )

        response_parts.append(
            f"\n*Ces informations proviennent de notre rapport interne sur l'analyse fondamentale financière "
            f"par approche Data Science (affichage des {min(3, len(results))} résultats les plus pertinents).*"
        )

        return "\n".join(response_parts)

    except Exception as e:
        return f"Erreur lors de l'accès au document de recherche : {str(e)}"

def search_financial_concepts(concept: str) -> str:
    """Fonction d'assistance pour rechercher des concepts financiers/science des données spécifiques"""
    financial_queries = {
        "ratios financiers": "ratios financiers analyse fondamentale",
        "modèles prédictifs": "modèles prédictifs machine learning finance",
        "analyse technique": "analyse technique indicateurs financiers",
        "données financières": "données financières sources preprocessing",
        "performance": "performance évaluation modèles financiers"
    }
    
    query = financial_queries.get(concept.lower(), concept)
    return query_research_document(query, max_results=4)