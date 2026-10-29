
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(page_title="Gaurang AI Portfolio - Real Meta AI", layout="wide")
st.title("🚀 Gaurang's AI Portfolio - Powered by Real Meta AI")

# --- REAL META AI (Llama 3.3) ---
def call_meta_ai(prompt):
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None, "Add GROQ_API_KEY in Streamlit Secrets: Manage App → Settings → Secrets → GROQ_API_KEY = \"gsk_...\""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 800
        }
        res = requests.post(url, headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content'], None
        else:
            return None, f"API Error: {res.text[:300]}"
    except Exception as e:
        return None, str(e)

# --- DATA FOR TAB 1 & 3 ---
lanes = [
    {"id":1, "name":"RAG", "desc":"Build knowledge base from docs, search with embeddings. Your Raga Finder is this.", "skills":"Vector DB, Embeddings", "money":"Low"},
    {"id":2, "name":"AI Apps Wrapper", "desc":"Wrap LLMs into useful tools like resume scanner, your 8 lanes app.", "skills":"Streamlit, API", "money":"Medium"},
    {"id":3, "name":"Automation", "desc":"Automate emails, leads, reports using AI agents that work while you sleep.", "skills":"Make, n8n, LangChain", "money":"High"},
    {"id":4, "name":"Evaluation", "desc":"Test if your AI is actually good.", "skills":"RAGAS", "money":"High"},
    {"id":5, "name":"Agentic AI", "desc":"AI agents that plan, use tools, browse.", "skills":"CrewAI", "money":"Very High"},
    {"id":6, "name":"Voice & Video", "desc":"Voice assistants, video generation.", "skills":"Whisper, ElevenLabs", "money":"High"},
    {"id":7, "name":"Fine-tuning", "desc":"Train your own small model.", "skills":"LoRA", "money":"Medium"},
    {"id":8, "name":"AI Products", "desc":"Full SaaS product with users, payments.", "skills":"Product", "money":"Very High"},
]
df_lanes = pd.DataFrame(lanes)
df_lanes['text'] = df_lanes['name'] + " " + df_lanes['desc']

def semantic_search(query, df):
    vectorizer = TfidfVectorizer().fit(df['text'].tolist() + [query])
    vectors = vectorizer.transform(df['text'].tolist() + [query])
    sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
    df['score'] = sims
    return df.sort_values('score', ascending=False).head(3)

tab1, tab2, tab3 = st.tabs(["🧠 8 Lanes", "🎵 Raga AI (No PDF)", "🎙️ Voice + Meta AI"])

# TAB 1
with tab1:
    st.header("🧠 What Do You Want To Build?")
    query = st.text_input("Type: make money while sleeping, voice clone, jarvis...", key="lane_search")
    if query:
        results = semantic_search(query, df_lanes.copy())
        for _, r in results.iterrows():
            st.success(f"**Lane {r['id']}: {r['name']}** ({r['score']:.0%} match)")
            st.write(f"{r['desc']} | Skills: {r['skills']} | Money: {r['money']}")

# TAB 2 - FINAL VERSION - NO PDF UPLOAD
with tab2:
    st.header("🎵 Raga Finder - Pure Meta AI")
    st.info("No PDF needed. Meta AI (Llama 3.3) knows 100+ Hindustani Ragas.")
    
    raga_name = st.text_input("Enter Raga Name:", placeholder="e.g. Yaman, Bhairav, Todi, Darbari, Bhairavi", key="raga_name_final")
    
    instruction = st.selectbox("What details do you want?", 
        ["Full Details: Aroha, Avaroha, Pakad, Thaat, Time, Vadi, Samvadi, Similar Ragas",
         "Only Aroha Avaroha for practice",
         "Beginner 2-line summary + mood",
         "Compare with 2 similar ragas",
         "Give bandish / composition idea + notation"],
        key="raga_inst_final"
    )
    
    if raga_name:
        prompt = f"""
You are Meta AI, a world-class Hindustani Classical Music Guru with 30 years of knowledge. You are accurate and authentic.

TASK: Provide details for Raga: {raga_name}
INSTRUCTION: {instruction}

STRICT FORMAT:
**Raga:** {raga_name}
**Thaat:** 
**Prahar / Time:** 
**Aroha:** 
**Avaroha:** 
**Pakad:** 
**Vadi / Samvadi:** 
**Jati:** 
**Mood / Rasa:** 
**Similar Ragas:** 
**Bandish / Extra:** (as per instruction)

RULES: Use proper Sargam (Sa Re Ga Ma Pa Dha Ni), correct wrong spellings, no hallucination.
"""
        if st.button(f"Get {raga_name} Details from Meta AI 🤖", key="get_raga_btn"):
            with st.spinner(f"Meta AI is recalling {raga_name}..."):
                answer, err = call_meta_ai(prompt)
                if answer:
                    st.success(f"Answer for {raga_name}")
                    st.markdown(answer)
                    safe = answer.replace("'", "").replace('"', "").replace("\n"," ")[:500]
                    st.markdown(f"""<button onclick="var m=new SpeechSynthesisUtterance('{safe}');window.speechSynthesis.speak(m);">▶️ Hear Details</button>""", unsafe_allow_html=True)
                else:
                    st.error(err)

# TAB 3
with tab3:
    st.header("🎙️ Voice + Meta AI Career Guide")
    st.write("Try: `make money while sleeping` or `Yaman raga` - Meta AI will answer and speak.")
    q3 = st.text_input("Your Question:", placeholder="e.g. I want to make money while sleeping", key="voice_final")
    inst3 = st.selectbox("Answer Style:", ["Career coach: give 3 steps to start", "Teacher: explain like I'm 10", "Investor: monetization potential"], key="inst3_final")
    
    if q3:
        results = semantic_search(q3, df_lanes.copy())
        context = f"Best matching Lane is {results.iloc[0]['name']}: {results.iloc[0]['desc']}"
        prompt3 = f"""
You are Meta AI, AI Career Guide and Raga Guru.
CONTEXT: {context}
USER QUESTION: {q3}
INSTRUCTION: {inst3}
If question is about raga, give Aroha Avaroha. If about career, map to Lane and give 3 actionable steps. Be concise.
"""
        if st.button("Ask Meta AI + Speak 🔊", key="voice_btn"):
            with st.spinner("Calling Meta AI..."):
                ans, err = call_meta_ai(prompt3)
                if ans:
                    st.success(ans)
                    safe = ans.replace("'", "").replace('"', "").replace("\n"," ")[:500]
                    st.markdown(f"""<button onclick="var m=new SpeechSynthesisUtterance('{safe}');window.speechSynthesis.speak(m);">▶️ Hear Meta AI</button>""", unsafe_allow_html=True)
                else:
                    st.error(err)
