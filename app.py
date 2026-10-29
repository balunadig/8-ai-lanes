
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="Gaurang AI Portfolio", layout="wide")
st.title("🚀 Gaurang's AI Portfolio - 2 Projects, 1 Link")

tab1, tab2 = st.tabs(["🧠 8 Lanes - Real AI", "🎵 Raga Finder - RAG"])

# ================= TAB 1: 8 LANES =================
with tab1:
    st.header("8 Lanes - Real AI Search")
    lanes = [
        {"id":1, "name":"RAG", "desc":"Build knowledge base from PDFs, search with embeddings, answer with LLM. Your Raga Finder.", "skills":"Vector DB, Embeddings", "money":"Low"},
        {"id":2, "name":"AI Apps Wrapper", "desc":"Wrap LLMs into useful tools like resume scanner, dashboard, your 8 lanes app.", "skills":"Streamlit, API", "money":"Medium"},
        {"id":3, "name":"Automation", "desc":"Automate emails, leads, reports using AI agents that work while you sleep.", "skills":"Make, n8n, LangChain", "money":"High"},
        {"id":4, "name":"Evaluation", "desc":"Test if your AI is actually good, measure accuracy.", "skills":"RAGAS", "money":"High"},
        {"id":5, "name":"Agentic AI", "desc":"AI agents that plan, use tools, browse, and complete tasks autonomously.", "skills":"CrewAI, AutoGen", "money":"Very High"},
        {"id":6, "name":"Voice & Video", "desc":"Voice assistants, video generation, avatars.", "skills":"Whisper, ElevenLabs", "money":"High"},
        {"id":7, "name":"Fine-tuning", "desc":"Train your own small model on your own data.", "skills":"LoRA", "money":"Medium"},
        {"id":8, "name":"AI Products", "desc":"Full SaaS product with users, payments, scaling.", "skills":"Product", "money":"Very High"},
    ]
    df = pd.DataFrame(lanes)
    df['text'] = df['name'] + " " + df['desc'] + " " + df['skills']
    
    query = st.text_input("What do you want to build? (Try: make money while sleeping)", key="lane_search")
    if query:
        vectorizer = TfidfVectorizer().fit(df['text'].tolist() + [query])
        vectors = vectorizer.transform(df['text'].tolist() + [query])
        sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
        df['score'] = sims
        results = df.sort_values('score', ascending=False).head(3)
        for _, r in results.iterrows():
            st.success(f"**Lane {r['id']}: {r['name']}** ({r['score']:.0%} match) - {r['desc']}")
    else:
        st.dataframe(df[['id','name','desc','money']], use_container_width=True)

    st.divider()
    goal = st.selectbox("Your goal?", ["Learn", "Make Money Fast", "Build Portfolio", "Get Job", "Build Startup"])
    def advise(g):
        if g=="Make Money Fast": return "Start Lane 3 Automation → Automate cold emails, charge $500/mo"
        if g=="Build Portfolio": return "You have Lane 1 & 2. Add Lane 3 + 6 for strong portfolio"
        if g=="Get Job": return "Focus Lane 1 RAG + Lane 4 Eval. Companies need accurate RAG"
        if g=="Build Startup": return "Lane 2 → 3 → 5. Your app is Lane 2, add automation, then agents"
        return "Master Lane 1 RAG deeply. Your Raga Finder is great start."
    if st.button("Get My AI Plan"):
        st.info(f"🤖 {advise(goal)}")
        st.balloons()

# ================= TAB 2: RAGA FINDER =================
with tab2:
    st.header("🎵 Raga Finder - Real RAG")
    st.write("Upload any PDF about Ragas → Ask questions → AI finds answer using same TF-IDF RAG")
    
    uploaded = st.file_uploader("Upload Raga PDF", type=["pdf"])
    
    if uploaded:
        reader = PyPDF2.PdfReader(uploaded)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # Chunking
        chunks = [text[i:i+500] for i in range(0, len(text), 400)]
        st.success(f"PDF loaded: {len(chunks)} chunks created (RAG Step 1 & 2 done)")
        
        q2 = st.text_input("Ask about Raga (e.g., What is Yaman raga aroha?)", key="raga_q")
        
        if q2 and chunks:
            vectorizer2 = TfidfVectorizer().fit(chunks + [q2])
            vecs = vectorizer2.transform(chunks + [q2])
            sims2 = cosine_similarity(vecs[-1], vecs[:-1])[0]
            best_idx = sims2.argmax()
            st.subheader(f"Best Match (Score: {sims2[best_idx]:.0%}) - This is RAG Retrieval:")
            st.info(chunks[best_idx][:1000])
            st.caption("Lesson: This is exactly RAG → Chunk → Embed (TF-IDF) → Similarity Search → Show answer")
    else:
        st.info("Upload a PDF to activate RAG. Try any music theory PDF you have.")
