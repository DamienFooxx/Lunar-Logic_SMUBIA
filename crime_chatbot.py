import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from typing import List, Dict, Tuple
import json
import logging
from transformers import pipeline
import requests

def ollama_generate(prompt: str, model: str = "llama2", max_tokens: int = 200) -> str:
    url = "http://localhost:11411/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "num_ctx": 2048,         # context size
        "num_gpu_layers": 10     # tune for performance on Apple Silicon
    }

    response = requests.post(url, json=payload, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama request failed: {response.text}")

    output_parts = []
    for chunk in response.iter_content(decode_unicode=True):
        if chunk:
            output_parts.append(chunk)

    return "".join(output_parts)



class CrimeDataRAG:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.generator = ollama_generate
        self.vector_store = None
        self.documents = []
        
    def load_and_process_data(self, nodes_path: str, edges_path: str, patterns_path: str,
                            risk_scores_path: str, feature_importance_path: str,
                            cleaned_data_path: str, relationships_path: str) -> None:
        try:
            # Load all datasets
            nodes_df = pd.read_csv(nodes_path)
            edges_df = pd.read_csv(edges_path)
            patterns_df = pd.read_csv(patterns_path)
            risk_scores_df = pd.read_csv(risk_scores_path)
            feature_importance_df = pd.read_csv(feature_importance_path)
            cleaned_data_df = pd.read_csv(cleaned_data_path)
            relationships_df = pd.read_csv(relationships_path)
            
            # Create documents for each entity
            for _, node in nodes_df.iterrows():
                doc = self._create_entity_document(
                    node, edges_df, patterns_df, risk_scores_df, 
                    relationships_df, cleaned_data_df
                )
                self.documents.append(doc)
                
            # Add feature importance information
            self.documents.append(self._create_feature_importance_doc(feature_importance_df))
            
            self._create_vector_store()
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def _create_entity_document(self, node: pd.Series, edges_df: pd.DataFrame, 
                              patterns_df: pd.DataFrame, risk_scores_df: pd.DataFrame,
                              relationships_df: pd.DataFrame, cleaned_data_df: pd.DataFrame) -> str:
        doc_parts = [
            f"Entity: {node['Entity']}",
            f"Type: {node['Type']}",
            f"Number of Crimes: {node['NumCrimes']}",
            f"Crimes: {node['Crimes']}"
        ]

        # Add risk score information
        risk_info = risk_scores_df[risk_scores_df['Entity'] == node['Entity']]
        if not risk_info.empty:
            doc_parts.extend([
                f"Risk Score: {risk_info.iloc[0]['RiskScore']:.2f}",
                f"Risk Percentile: {risk_info.iloc[0]['RiskPercentile']:.2f}",
                f"Risk Level: {risk_info.iloc[0]['RiskLevel']}"
            ])

        # Add relationships
        entity_edges = edges_df[
            (edges_df['Source'] == node['Entity']) | 
            (edges_df['Target'] == node['Entity'])
        ]
        if not entity_edges.empty:
            doc_parts.append("\nRelationships:")
            for _, edge in entity_edges.iterrows():
                other_entity = edge['Target'] if edge['Source'] == node['Entity'] else edge['Source']
                doc_parts.append(
                    f"- Connected to {other_entity} through {edge['Relationship']} "
                    f"(Crime: {edge['CrimeType']}, Evidence: {edge['EvidenceStrength']})"
                )

        # Add detailed relationships
        entity_relationships = relationships_df[
            (relationships_df['subject'] == node['Entity']) |
            (relationships_df['object'] == node['Entity'])
        ]
        if not entity_relationships.empty:
            doc_parts.append("\nDetailed Crime Relationships:")
            for _, rel in entity_relationships.iterrows():
                doc_parts.append(
                    f"- {rel['subject']} -> {rel['predicate']} -> {rel['object']} "
                    f"(Crime: {rel['crime_type']}, Evidence: {rel['evidence_strength']})"
                )

        # Add patterns
        entity_patterns = patterns_df[patterns_df['Entity'] == node['Entity']]
        if not entity_patterns.empty:
            doc_parts.append("\nCrime Patterns:")
            for _, pattern in entity_patterns.iterrows():
                doc_parts.append(
                    f"- Pattern: {pattern['CrimeType']}, "
                    f"Centrality: {pattern['Centrality']:.3f}"
                )

        return "\n".join(doc_parts)

    def _create_feature_importance_doc(self, feature_importance_df: pd.DataFrame) -> str:
        doc_parts = ["Feature Importance Analysis:"]
        for _, row in feature_importance_df.iterrows():
            doc_parts.append(f"- {row['feature']}: {row['importance']:.3f}")
        return "\n".join(doc_parts)

    def _create_vector_store(self) -> None:
        embeddings = self.embed_model.encode(self.documents)
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(np.array(embeddings).astype('float32'))

    def get_relevant_context(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.embed_model.encode([query])
        D, I = self.vector_store.search(
            np.array(query_embedding).astype('float32'), k
        )
        return [self.documents[i] for i in I[0]]

    def generate_response(self, query: str, context: List[str]) -> str:
        prompt = f"""Based on the crime network analysis:
    {' '.join(context)}

    Question: {query}
    Analysis:"""
        
        try:
            raw_text = self.generator(prompt, model="llama2", max_tokens=300)
            return raw_text.split("Analysis:")[-1].strip()
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "Error generating response. Relevant context:\n\n" + "\n".join(context)

def create_streamlit_app():
    st.title("Crime Network Analysis Chatbot")
    
    @st.cache_resource
    def load_rag_system():
        try:
            rag = CrimeDataRAG()
            base_path = "/Users/damienfoo/Desktop/SMUBIA Datathon Lunar Logic/FINAL FINAL PLEASE/Data"
            tableau_path = f"{base_path}/Tablaeu Data"
            
            rag.load_and_process_data(
                f"{tableau_path}/crime_network_clean_nodes.csv",
                f"{tableau_path}/crime_network_clean_edges.csv",
                f"{tableau_path}/crime_network_clean_patterns.csv",
                f"{tableau_path}/entity_risk_scores.csv",
                f"{tableau_path}/feature_importance.csv",
                f"{base_path}/process2_cleaned.csv",
                f"{base_path}/process3_crime_relationships_enhanced.csv"
            )
            return rag
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            return None

    rag = load_rag_system()
    
    if not rag:
        st.error("Failed to initialize the system. Please check the logs.")
        return

    query = st.text_input("Ask a question about the crime network:")
    
    if query:
        try:
            with st.spinner("Searching relevant information..."):
                context = rag.get_relevant_context(query)
            
            with st.spinner("Generating response..."):
                response = rag.generate_response(query, context)
                
            st.write("Response:", response)
            
            with st.expander("View Source Context"):
                for i, doc in enumerate(context, 1):
                    st.text(f"Document {i}:\n{doc}\n")
                    
        except Exception as e:
            st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_streamlit_app()