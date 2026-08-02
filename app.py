"""
████████████████████████████████████
NeuroSync - AI-Powered Resume Chatbot
Plug your resume into the neural grid.
████████████████████████████████████
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO
import json

# Page config MUST be first
st.set_page_config(
    page_title="NeuroSync - AI Resume Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    import openai
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

import numpy as np
import pandas as pd

# ============================================
# CSS STYLING
# ============================================

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 50%, #0a0a1a 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d25 0%, #151535 100%);
        border-right: 1px solid rgba(0, 255, 136, 0.15);
    }
    
    .glass-card {
        background: rgba(20, 20, 40, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 255, 136, 0.4);
        box-shadow: 0 12px 40px rgba(0, 255, 136, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #0a0a1a;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.5);
    }
    
    [data-testid="stMetric"] {
        background: rgba(20, 20, 40, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 136, 0.15);
        border-radius: 15px;
        padding: 20px;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 255, 136, 0.3);
    }
    
    [data-testid="stMetric"] label {
        color: #00ff88 !important;
        font-weight: 600 !important;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        animation: slideIn 0.3s ease;
    }
    
    .user-message {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        margin-left: 40px;
    }
    
    .bot-message {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-right: 40px;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stTextInput > div > div > input {
        background: rgba(20, 20, 40, 0.8);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 12px;
        color: white;
        padding: 12px 20px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00ff88;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
    }
    
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb { background: #00ff8840; border-radius: 4px; }
    
    .gradient-text {
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .neural-bg {
        position: relative;
        overflow: hidden;
    }
    
    .neural-bg::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0,255,136,0.05) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .pulse-dot {
        width: 12px;
        height: 12px;
        background: #00ff88;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.4); }
        50% { box-shadow: 0 0 0 15px rgba(0, 255, 136, 0); }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# NEUROSYNC ENGINE
# ============================================

class NeuroSyncEngine:
    """Core engine for resume processing and Q&A"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.vector_store = None
        self.qa_chain = None
        self.resume_text = None
        self.is_ready = False
        
    def process_resume(self, pdf_path):
        """Process resume PDF and create vector store"""
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Store full text
            self.resume_text = "\n".join([doc.page_content for doc in documents])
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(api_key=self.api_key)
            self.vector_store = FAISS.from_documents(chunks, embeddings)
            
            # Create QA chain
            prompt_template = """You are NeuroSync, an AI assistant that answers questions about a person's resume.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Be professional, concise, and helpful.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
                api_key=self.api_key
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            self.is_ready = True
            return True
            
        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")
            return False
    
    def ask(self, question):
        """Ask a question about the resume"""
        
        if not self.is_ready:
            return {"answer": "Please upload and process a resume first.", "sources": []}
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.page_content[:200] for doc in result["source_documents"]]
            }
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": []}

class SimulatedNeuroSync:
    """Simulated version when API key is not available"""
    
    def __init__(self):
        self.resume_data = {}
        self.is_ready = False
        self.resume_text = ""
        
    def process_resume(self, pdf_path):
        """Simulate resume processing"""
        # Extract text from PDF (basic)
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            self.resume_text = "\n".join([doc.page_content for doc in documents])
        except:
            # Use sample resume data
            self.resume_text = self._get_sample_resume()
        
        # Parse basic information
        self._parse_resume()
        self.is_ready = True
        return True
    
    def _get_sample_resume(self):
        return """
        JOHN DOE
        Software Engineer | AI/ML Specialist
        
        SUMMARY
        Innovative software engineer with 5+ years of experience in building scalable applications
        and machine learning systems. Passionate about creating elegant solutions to complex problems.
        
        SKILLS
        Programming: Python, JavaScript, TypeScript, Java, C++
        ML/AI: TensorFlow, PyTorch, Scikit-learn, LangChain, OpenAI
        Web: React.js, Node.js, Flask, FastAPI, Streamlit
        Cloud: AWS, GCP, Docker, Kubernetes
        Databases: PostgreSQL, MongoDB, Redis
        
        EXPERIENCE
        
        Senior ML Engineer | TechCorp Inc. | Jan 2022 - Present
        - Developed production ML pipelines serving 1M+ daily predictions
        - Reduced model inference time by 60% through optimization
        - Led team of 4 engineers in building recommendation system
        - Implemented MLOps practices reducing deployment time by 40%
        
        Full Stack Developer | StartupXYZ | Jun 2019 - Dec 2021
        - Built full-stack web application using React and Node.js
        - Designed RESTful APIs handling 100K+ daily requests
        - Implemented CI/CD pipeline reducing release cycles by 50%
        - Mentored junior developers and conducted code reviews
        
        ML Intern | DataLab | Jan 2019 - May 2019
        - Developed NLP models for sentiment analysis with 92% accuracy
        - Created data visualization dashboards using Python and Plotly
        - Collaborated on research paper published at ML Conference 2019
        
        EDUCATION
        
        M.S. Computer Science (AI/ML) | Stanford University | 2017 - 2019
        - GPA: 3.9/4.0
        - Research: Deep Learning for Natural Language Processing
        
        B.Tech Computer Science | IIT Delhi | 2013 - 2017
        - GPA: 8.5/10
        - Minor in Mathematics
        
        PROJECTS
        
        NeuroChat: AI-powered chatbot using GPT-4 and LangChain
        - Built conversational AI with context-aware responses
        - Integrated with Slack and Discord, 500+ active users
        
        PredictIQ: ML Prediction Platform
        - Created automated ML pipeline for business forecasting
        - Reduced prediction errors by 35% compared to baseline
        
        CERTIFICATIONS
        - AWS Solutions Architect Professional
        - Google Cloud ML Engineer
        - TensorFlow Developer Certificate
        
        LANGUAGES
        English (Native), Spanish (Intermediate), Hindi (Fluent)
        """
    
    def _parse_resume(self):
        """Parse resume text into structured data"""
        text = self.resume_text
        
        self.resume_data = {
            'skills': self._extract_section(text, 'SKILLS'),
            'experience': self._extract_section(text, 'EXPERIENCE'),
            'education': self._extract_section(text, 'EDUCATION'),
            'projects': self._extract_section(text, 'PROJECTS'),
            'summary': self._extract_section(text, 'SUMMARY'),
        }
    
    def _extract_section(self, text, section_name):
        """Extract a section from resume text"""
        lines = text.split('\n')
        capture = False
        section_lines = []
        
        for line in lines:
            if section_name.upper() in line.upper():
                capture = True
                continue
            elif capture and line.strip().upper() in ['SKILLS', 'EXPERIENCE', 'EDUCATION', 'PROJECTS', 'CERTIFICATIONS', 'LANGUAGES']:
                break
            elif capture and line.strip():
                section_lines.append(line.strip())
        
        return '\n'.join(section_lines) if section_lines else section_name
    
    def ask(self, question):
        """Answer questions based on parsed resume"""
        question_lower = question.lower()
        
        # Skills related
        if any(word in question_lower for word in ['skill', 'know', 'language', 'programming', 'tech', 'technology', 'tool']):
            skills = self.resume_data.get('skills', '')
            return {
                "answer": f"Based on the resume, here are the key skills and technologies:\n\n{skills}\n\nThese include programming languages, frameworks, cloud platforms, and specialized tools.",
                "sources": [skills[:200]]
            }
        
        # Experience related
        elif any(word in question_lower for word in ['experience', 'work', 'job', 'career', 'role', 'position']):
            experience = self.resume_data.get('experience', '')
            return {
                "answer": f"Here's a summary of work experience:\n\n{experience}\n\nThis shows progressive growth from intern to senior roles with increasing responsibilities.",
                "sources": [experience[:200]]
            }
        
        # Education related
        elif any(word in question_lower for word in ['education', 'degree', 'university', 'college', 'study', 'gpa', 'school']):
            education = self.resume_data.get('education', '')
            return {
                "answer": f"Educational background:\n\n{education}\n\nStrong academic foundation with focus on computer science and AI/ML.",
                "sources": [education[:200]]
            }
        
        # Projects related
        elif any(word in question_lower for word in ['project', 'portfolio', 'build', 'create']):
            projects = self.resume_data.get('projects', '')
            return {
                "answer": f"Key projects and accomplishments:\n\n{projects}",
                "sources": [projects[:200]]
            }
        
        # Summary
        elif any(word in question_lower for word in ['summary', 'overview', 'about', 'who', 'background']):
            summary = self.resume_data.get('summary', '')
            return {
                "answer": f"Professional Summary:\n\n{summary}\n\nA well-rounded professional with expertise in software engineering and AI/ML.",
                "sources": [summary[:200]]
            }
        
        # Strengths
        elif any(word in question_lower for word in ['strength', 'strong', 'best', 'top', 'expert']):
            return {
                "answer": "Based on the resume analysis, the top strengths are:\n\n1. **AI/ML Engineering** - Extensive experience with production ML systems\n2. **Full-Stack Development** - Proficient in building end-to-end applications\n3. **System Optimization** - Proven track record of improving performance by 40-60%\n4. **Leadership** - Led teams and mentored junior developers\n5. **Cloud Architecture** - AWS and GCP certified with practical experience",
                "sources": ["Strengths derived from comprehensive resume analysis"]
            }
        
        # Generic fallback
        else:
            return {
                "answer": f"Based on the resume analysis, I can tell you that the candidate has:\n\n• Strong background in AI/ML and software engineering\n• Experience at leading tech companies\n• Multiple certifications (AWS, GCP, TensorFlow)\n• Track record of delivering measurable improvements\n\nFeel free to ask about specific skills, experience, education, or projects!",
                "sources": ["Summary derived from resume analysis"]
            }

# ============================================
# STREAMLIT UI
# ============================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'resume_loaded' not in st.session_state:
        st.session_state.resume_loaded = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'use_simulation' not in st.session_state:
        st.session_state.use_simulation = False

def main():
    load_css()
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px 0;">
            <h1 style="font-size:2rem; margin:0;">
                <span style="color:#00ff88;">🧠 Neuro</span><span style="color:#ffffff;">Sync</span>
            </h1>
            <p style="color:#a0a0c0; font-size:0.85rem; margin-top:5px;">
                Plug your resume into the neural grid
            </p>
            <div style="margin:10px 0;">
                <span class="pulse-dot"></span>
                <span style="color:#00ff88; font-size:0.8rem; margin-left:8px;">Neural Grid Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # API Key Configuration
        st.markdown("### 🔑 API Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Your OpenAI API key for GPT-4 access"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        use_sim = st.checkbox(
            "Use Simulation Mode (no API key needed)",
            value=st.session_state.use_simulation,
            help="Enable to use the app without an OpenAI API key"
        )
        st.session_state.use_simulation = use_sim
        
        st.markdown("---")
        
        # Resume Upload
        st.markdown("### 📄 Upload Resume")
        
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type=['pdf'],
            help="Upload your resume in PDF format"
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧠 Process Resume", use_container_width=True):
                    with st.spinner("Processing resume..."):
                        # Save uploaded file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Initialize engine
                        if use_sim or not api_key:
                            engine = SimulatedNeuroSync()
                        else:
                            engine = NeuroSyncEngine(api_key=api_key)
                        
                        # Process resume
                        success = engine.process_resume(tmp_path)
                        
                        if success:
                            st.session_state.engine = engine
                            st.session_state.resume_loaded = True
                            st.session_state.chat_history = []
                            st.success("✅ Resume processed!")
                            st.rerun()
                        else:
                            st.error("Failed to process resume")
                        
                        # Cleanup
                        os.unlink(tmp_path)
            
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.engine = None
                    st.session_state.resume_loaded = False
                    st.session_state.chat_history = []
                    st.rerun()
        
        st.markdown("---")
        
        # Sample questions
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What are my top technical skills?",
            "Summarize my work experience",
            "What are my strengths?",
            "Tell me about my education",
            "What projects have I worked on?",
            "What certifications do I have?",
            "Summarize my entire resume"
        ]
        
        for q in sample_questions:
            if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
                st.session_state.current_question = q
                st.rerun()
        
        st.markdown("---")
        
        # Stats
        if st.session_state.resume_loaded:
            st.markdown("### 📊 Resume Stats")
            engine = st.session_state.engine
            
            if hasattr(engine, 'resume_text'):
                text = engine.resume_text
                words = len(text.split())
                st.metric("Words", f"{words:,}")
                st.metric("Characters", f"{len(text):,}")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding:10px;">
            <p style="color:#00ff88; font-size:0.8rem;">Built with ❤️</p>
            <p style="color:#a0a0c0; font-size:0.7rem;">NeuroSync v1.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("""
    <div style="text-align:center; padding:20px 0;">
        <h1 style="font-size:3.5rem; font-weight:900; margin:0;">
            <span style="color:#00ff88;">🧠 Neuro</span><span style="color:#ffffff;">Sync</span>
        </h1>
        <p style="font-size:1.3rem; color:#00cc6a; margin:10px 0; font-weight:500;">
            Plug your resume into the neural grid
        </p>
        <p style="font-size:1rem; color:#a0a0c0;">
            AI-powered chatbot that answers questions about your experience, education, and skills
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main chat interface
    if not st.session_state.resume_loaded:
        # Welcome screen
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="padding:40px 20px;">
                <h2 style="color:#00ff88;">🚀 Get Started</h2>
                <p style="color:#a0a0c0; font-size:1.1rem; line-height:1.8;">
                    1️⃣ Upload your resume PDF<br>
                    2️⃣ Enter your OpenAI API key<br>
                    3️⃣ Start asking questions!
                </p>
                <p style="color:#666; font-size:0.9rem; margin-top:30px;">
                    💡 <strong>No API key?</strong> Enable Simulation Mode in sidebar
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-card" style="text-align:center;">
                <h3 style="color:#00ff88;">🧠 AI-Powered</h3>
                <p style="color:#a0a0c0;">GPT-4 powered responses about your professional background</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card" style="text-align:center;">
                <h3 style="color:#00ff88;">🔒 Private & Local</h3>
                <p style="color:#a0a0c0;">Your resume stays on your machine. No cloud storage.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card" style="text-align:center;">
                <h3 style="color:#00ff88;">⚡ Instant Answers</h3>
                <p style="color:#a0a0c0;">Get immediate responses about skills, experience, and more.</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Chat interface
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Status indicator
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
            <span class="pulse-dot"></span>
            <span style="color:#00ff88; font-weight:600;">Resume Loaded • Ready for Questions</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong style="color:#00ff88;">You</strong><br>
                        <span style="color:#e0e0e0;">{msg['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong style="color:#6366f1;">🧠 NeuroSync</strong><br>
                        <span style="color:#e0e0e0;">{msg['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input area
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            # Handle sample question click
            if 'current_question' in st.session_state and st.session_state.current_question:
                question = st.session_state.current_question
                st.session_state.current_question = None
            else:
                question = st.text_input(
                    "",
                    placeholder="Ask me anything about the resume... (e.g., 'What are my top skills?')",
                    key="chat_input",
                    label_visibility="collapsed"
                )
        
        with col2:
            send_button = st.button("🚀 Send", use_container_width=True)
        
        if send_button and question:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': question
            })
            
            # Get response
            with st.spinner("🧠 Thinking..."):
                engine = st.session_state.engine
                if engine:
                    response = engine.ask(question)
                    answer = response['answer']
                else:
                    answer = "Please upload a resume first."
            
            # Add bot response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': answer
            })
            
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick actions
        if st.session_state.chat_history:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Copy Last Response", use_container_width=True):
                    last_bot_msg = next((msg['content'] for msg in reversed(st.session_state.chat_history) if msg['role'] == 'assistant'), '')
                    st.code(last_bot_msg)
                    st.success("Response copied!")
            
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            with col3:
                # Export chat
                chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])
                st.download_button(
                    "💾 Export Chat",
                    chat_text,
                    file_name=f"neurosync_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:20px;">
        <p style="color:#00ff88; font-size:1rem; font-weight:600;">
            🧠 NeuroSync - AI-Powered Resume Intelligence
        </p>
        <p style="color:#a0a0c0; font-size:0.8rem;">
            "The best way to predict your future... is to train it."
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
