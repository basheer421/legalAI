import streamlit as st
from dataclasses import dataclass
import openai
from typing import List
import PyPDF2
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Response:
    content: str
    messages: List[Message]

from langchain.embeddings.base import Embeddings

class LightEmbeddings(Embeddings):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

class SambanovaChat:
    def __init__(self, model="Meta-Llama-3.1-70B-Instruct", temperature=0.1):
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(
            api_key=os.environ["SAMBANOVA_API_KEY"],
            base_url="https://api.sambanova.ai/v1"
        )
    
    def complete(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=0.1
        )
        return response.choices[0].message.content

class Agent:
    def __init__(self, name, role, model, team=None, instructions=None, vector_store=None):
        self.name = name
        self.role = role
        self.model = model
        self.team = team
        self.instructions = instructions
        self.vector_store = vector_store

    def run(self, query):
        document_content = st.session_state.get('document_content', '')
        context = ""
        if self.vector_store and query:
            relevant_docs = self.vector_store.similarity_search(query, k=3)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
        
        system_prompt = f"""You are {self.name}, a {self.role}.
        Instructions: {'; '.join(self.instructions) if self.instructions else ''}
        
        Document Content:
        {document_content}
        
        Relevant Context:
        {context}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = self.model.complete(messages)
        return Response(content=response, messages=[Message(role="assistant", content=response)])

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    st.title("AI Legal Document Analyzer 👨‍⚖️")

    knowledge_base_path = "knowledge"
    try:
        knowledge_base_path = os.path.join(os.path.dirname(__file__), "knowledge")
        if os.path.exists(knowledge_base_path):
            loader = DirectoryLoader(knowledge_base_path, glob="*.txt", loader_cls=TextLoader)
            documents = loader.load()
            embeddings = LightEmbeddings()
            vector_store = FAISS.from_documents(documents, embeddings)
            st.session_state.vector_store = vector_store
            st.sidebar.success(f"✅ Loaded {len(documents)} knowledge base files")
        else:
            st.warning(f"Knowledge directory not found at: {knowledge_base_path}")
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")

    uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf'])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                
                st.session_state.document_content = pdf_text
                model = SambanovaChat()
                vector_store = st.session_state.get('vector_store')
                
                legal_researcher = Agent(
                    name="Legal Researcher",
                    role="Legal research specialist",
                    model=model,
                    vector_store=vector_store,
                    instructions=[
                        "Find and cite relevant legal cases and precedents",
                        "Provide detailed research summaries"
                    ]
                )

                contract_analyst = Agent(
                    name="Contract Analyst",
                    role="Contract analysis specialist",
                    model=model,
                    vector_store=vector_store,
                    instructions=[
                        "Review contracts thoroughly",
                        "Identify key terms and potential issues"
                    ]
                )

                legal_strategist = Agent(
                    name="Legal Strategist", 
                    role="Legal strategy specialist",
                    model=model,
                    vector_store=vector_store,
                    instructions=[
                        "Develop comprehensive legal strategies",
                        "Provide actionable recommendations"
                    ]
                )

                st.session_state.legal_team = Agent(
                    name="Legal Team Lead",
                    role="Legal team coordinator",
                    model=model,
                    vector_store=vector_store,
                    team=[legal_researcher, contract_analyst, legal_strategist],
                    instructions=[
                        "Coordinate analysis between team members",
                        "Provide comprehensive responses"
                    ]
                )
                
                st.success("✅ Document processed and team initialized!")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

        if 'legal_team' in st.session_state:
            st.subheader("Document Analysis")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Contract Review", "Legal Research", "Risk Assessment", "Compliance Check", "Custom Query"]
            )

            if analysis_type == "Custom Query":
                query = st.text_area("Enter your query:")
            else:
                queries = {
                    "Contract Review": "Review this contract and identify key terms and issues.",
                    "Legal Research": "Research relevant cases and precedents.",
                    "Risk Assessment": "Analyze potential legal risks and liabilities.",
                    "Compliance Check": "Check for regulatory compliance issues."
                }
                query = queries[analysis_type]

            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    response = st.session_state.legal_team.run(query)
                    st.markdown(response.content)

if __name__ == "__main__":
    main()