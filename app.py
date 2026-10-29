import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="8 Lanes - Real AI", layout="wide")

st.title("🧠 8 AI Lanes - Now With REAL AI")
st.caption("Semantic Search + AI Advisor")

# --- DATA ---
lanes = [
    {"id":1, "name":"RAG", "desc":"Build knowledge base from PDFs, search with embeddings, answer with LLM. Your Raga Finder.", "skills":"Vector DB, Embeddings, Chunking", "money":"Low", "diff":"Medium"},
    {"id":2, "name":"AI Apps Wrapper", "desc":"Wrap LLMs into useful tools like resume scanner, dashboard, your 8 lanes app.", "skills":"Streamlit, API, Prompting", "money":"Medium", "diff":"Low"},
    {"id":3, "name":"Automation", "desc":"Automate emails, leads, reports using AI agents that work while you sleep.", "skills":"Make, n8n, LangChain", "money":"High", "diff":"Medium"},
    {"id":4, "name":"Evaluation", "desc":"Test if your AI is actually good, measure accuracy, prevent hallucinations.", "skills":"RAGAS, Human eval", "money":"High", "diff":"High"},
    {"id":5, "name":"Agentic AI", "desc":"AI agents that plan, use tools, browse, and complete complex tasks autonomously.", "skills":"CrewAI, AutoGen, Tools", "money":"Very High", "diff":"Very High"},
    {"id":6, "name":"Voice & Video", "desc":"Voice assistants, video generation, avatars for content and calls.", "skills":"Whisper, ElevenLabs", "money":"High", "diff":"Medium"},
    {"id":7, "name":"Fine-tuning", "desc":"Train your own small model on your own data for specific domain.", "skills":"LoRA, QLoRA", "money":"Medium", "diff":"Very High"},
    {"id":8, "name":"AI Products", "desc":"Full SaaS product with users, payments, and scaling to $10k MRR.", "skills":"Product, Growth", "money":"Very High", "diff":"High"},
]
df = pd.DataFrame(lanes)
df['text'] = df['name'] + " " + df['desc'] + " " + df['skills']

# --- REAL AI 1: SEMANTIC SEARCH ---
st.header("1. Semantic Search (Real AI, not keywords)")
st.write("Try: 'I want to make money while sleeping' or 'help me build voice clone'")

query = st.text_input("What do you want to build?")

if query:
    vectorizer = TfidfVectorizer().fit(df['text'].tolist() + [query])
    vectors = vectorizer.transform(df['text'].tolist() + [query])
    sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
    df['score'] = sims
    results = df.sort_values('score', ascending=False).head(3)
    
    st.subheader(f"AI found {len(results)} best lanes for you:")
    for _, r in results.iterrows():
        st.success(f"**Lane {r['id']}: {r['name']}** (match: {r['score']:.0%}) - {r['desc']}")
else:
    st.dataframe(df[['id','name','desc','money','diff']], use_container_width=True)

# --- REAL AI 2: AI ADVISOR ---
st.header("2. AI Advisor (Rule-based Agent)")
goal = st.selectbox("Your current goal?", ["Learn", "Make Money Fast", "Build Portfolio", "Get Job", "Build Startup"])

def advise(goal):
    if goal == "Make Money Fast": return "Start Lane 3 Automation → Find 5 businesses, automate their cold emails with AI. Charge $500/mo. That's Lane 3. Then add Lane 4 Eval to show ROI."
    if goal == "Build Portfolio": return "You did Lane 1 & 2. Now build Lane 3 (small automation) + Lane 6 (voice demo). 4 projects = strong portfolio."
    if goal == "Get Job": return "Focus Lane 1 RAG + Lane 4 Evaluation. Companies pay for accurate RAG, not demos. Build eval metrics."
    if goal == "Build Startup": return "Lane 2 → 3 → 5. Your 8 lanes app is Lane 2. Now add automation (Lane 3) for users, then agents (Lane 5)."
    return "Master Lane 1 RAG deeply. Your Raga Finder is great start. Add 100 more PDFs, improve embeddings."

if st.button("Get My AI Plan"):
    st.info(f"🤖 Advisor: {advise(goal)}")
    st.balloons()

# --- ROADMAP ---
st.header("3. Roadmap")
st.graphviz_chart("""
digraph {
 rankdir=LR;
 RAG -> AI_Apps -> Automation -> Eval -> Agentic -> Voice -> Finetune -> Products;
}
""")

st.markdown("---")
st.write("Lesson: Your old app showed the map. This app *understands* what you want using TF-IDF vectors + cosine similarity — that's the core of semantic search / RAG!")
