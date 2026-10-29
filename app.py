import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="Gaurang AI Portfolio", layout="wide")
st.title("🚀 Gaurang's AI Portfolio - 3 Projects, 1 Link")

tab1, tab2, tab3 = st.tabs(["🧠 8 Lanes AI", "🎵 Raga Finder RAG", "🎙️ Voice Search - NEW"])

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
df_lanes = pd.DataFrame(lanes)
df_lanes['text'] = df_lanes['name'] + " " + df_lanes['desc'] + " " + df_lanes['skills']

def semantic_search(query, df):
    vectorizer = TfidfVectorizer().fit(df['text'].tolist() + [query])
    vectors = vectorizer.transform(df['text'].tolist() + [query])
    sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
    df['score'] = sims
    return df.sort_values('score', ascending=False).head(3)

# TAB 1
with tab1:
    st.header("8 Lanes - Real AI Search")
    query = st.text_input("What do you want to build? (Try: make money while sleeping)", key="lane_search")
    if query:
        results = semantic_search(query, df_lanes.copy())
        for _, r in results.iterrows():
            st.success(f"**Lane {r['id']}: {r['name']}** ({r['score']:.0%} match) - {r['desc']}")
    else:
        st.dataframe(df_lanes[['id','name','desc','money']], use_container_width=True)
    goal = st.selectbox("Your goal?", ["Learn", "Make Money Fast", "Build Portfolio", "Get Job", "Build Startup"])
    if st.button("Get My AI Plan"):
        plans = {
            "Make Money Fast": "Start Lane 3 Automation → Automate cold emails, charge $500/mo",
            "Build Portfolio": "You have Lane 1 & 2. Add Lane 3 + 6 for strong portfolio",
            "Get Job": "Focus Lane 1 RAG + Lane 4 Eval. Companies need accurate RAG",
            "Build Startup": "Lane 2 → 3 → 5. Your app is Lane 2, add automation, then agents",
            "Learn": "Master Lane 1 RAG deeply. Your Raga Finder is great start."
        }
        st.info(f"🤖 {plans[goal]}")
        st.balloons()

# TAB 2
with tab2:
    st.header("🎵 Raga Finder - Real RAG")
    uploaded = st.file_uploader("Upload Raga PDF", type=["pdf"], key="raga")
    if uploaded:
        reader = PyPDF2.PdfReader(uploaded)
        text = "".join([p.extract_text() or "" for p in reader.pages])
        chunks = [text[i:i+500] for i in range(0, len(text), 400)]
        st.success(f"PDF loaded: {len(chunks)} chunks created")
        q2 = st.text_input("Ask about Raga (e.g., What is Yaman aroha?)", key="raga_q")
        if q2 and chunks:
            v = TfidfVectorizer().fit(chunks + [q2])
            vecs = v.transform(chunks + [q2])
            sims2 = cosine_similarity(vecs[-1], vecs[:-1])[0]
            best_idx = sims2.argmax()
            st.info(f"**Best Match ({sims2[best_idx]:.0%}):** {chunks[best_idx][:1000]}")

# TAB 3 - NEW
with tab3:
    st.header("🎙️ Voice Search - Lane 6")
    st.write("This is Lane 6 Voice AI. Speak what you want to build, AI will find your lane + speak back.")
    
    st.info("💡 Lesson: Voice AI = Speech to Text (Whisper) → Semantic Search (Tab 1) → Text to Speech (ElevenLabs). You are building same pipeline!")
    
    # Option 1: Record directly (works on phone)
    audio = st.audio_input("Tap mic and say: I want to make money while sleeping")
    
    query_voice = st.text_input("Or type what you said:", placeholder="e.g. I want voice clone", key="voice_text")
    
    # If audio recorded, we show mock transcription (since Whisper needs API, we teach concept)
    if audio:
        st.audio(audio)
        st.success("Audio captured! (In production, Whisper would transcribe this to text)")
        if not query_voice:
            query_voice = st.text_input("Whisper heard (type it for now):", key="voice_mock")
    
    if query_voice:
        st.write(f"You said: **{query_voice}**")
        results = semantic_search(query_voice, df_lanes.copy())
        for _, r in results.iterrows():
            st.success(f"🎙️ Voice Result → **Lane {r['id']}: {r['name']}** ({r['score']:.0%}) - {r['desc']}")
        
        st.subheader("🔊 AI would now speak this answer (Text to Speech)")
        st.caption("In production: Use ElevenLabs API or gTTS to convert this text to voice. For now, your browser can read it.")
        
        # Simple browser TTS using HTML
        tts_text = f"For your goal {query_voice}, I recommend Lane {results.iloc[0]['id']} {results.iloc[0]['name']}. {results.iloc[0]['desc']}"
        st.markdown(f"""
        <button onclick="var msg = new SpeechSynthesisUtterance('{tts_text}'); window.speechSynthesis.speak(msg);">▶️ Click to Hear AI Speak Answer</button>
        """, unsafe_allow_html=True)
