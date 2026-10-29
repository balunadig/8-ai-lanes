import streamlit as st

st.set_page_config(page_title="8 Lanes incl Agentic AI", layout="wide")
st.title("🛣️ 8 AI Lanes = 7 Practical + Agentic AI")
st.caption("Search-proof like Kalashri/Kalashree: type 'agent', 'rag', 'lora', 'automate'")

LANES = {
    "rag": {
        "title": "1. RAG / Knowledge Assistants",
        "variants": ["rag", "knowledge", "chat docs", "pdf", "youtube", "langchain", "vector"],
        "hook": "Chat over your docs, ragas, PDFs, YouTube transcripts. LangChain + vector DB.",
        "stack": ["LangChain, Chroma/Pinecone", "Embeddings"],
        "projects": ["Raga Finder", "PDF Bandish Q&A"],
        "demand": "🔥 Very High", "money": "High", "learn": "3-4w", "badge": "YOU ARE HERE"
    },
    "ai apps": {
        "title": "2. AI Apps / Wrappers",
        "variants": ["wrappers", "ai apps", "summarizer", "translator", "streamlit", "next.js"],
        "hook": "Not agents — just smart UI: summarizers, translators, music recommenders, meditation timers with AI picks. Ship fast.",
        "stack": ["Streamlit / Next.js", "OpenAI API"],
        "projects": ["Meditation Timer + AI Raga", "Raga → Playlist"],
        "demand": "🔥 Highest Ship", "money": "Fastest $$", "learn": "2-3w", "badge": "RECOMMENDED NEXT"
    },
    "fine-tuning": {
        "title": "3. Fine-tuning / LoRA",
        "variants": ["fine-tuning", "finetuning", "lora", "hugging face", "alaap"],
        "hook": "Teach small model your style — e.g. model that writes alaap descriptions, or classifies ragas. Free on HF.",
        "stack": ["Hugging Face", "LoRA, Unsloth, Ollama"],
        "projects": ["Raga Classifier", "Alaap Generator"],
        "demand": "🔥 Specialist", "money": "Specialist pay", "learn": "4-5w", "badge": ""
    },
    "automation": {
        "title": "4. Automation (No Agents Needed)",
        "variants": ["automation", "n8n", "zapier", "make", "transcribe", "tag"],
        "hook": "n8n, Zapier, Make + AI: auto-transcribe concerts, tag ragas, build playlists. Less hype, more money.",
        "stack": ["n8n (free)", "Zapier/Make", "Whisper"],
        "projects": ["Auto-Transcribe → Notion", "Auto Playlist Builder"],
        "demand": "💰 Most $$", "money": "SMEs pay cash", "learn": "1-2w", "badge": "FAST MONEY"
    },
    "eval": {
        "title": "5. Eval / Guardrails / Prompt Eng",
        "variants": ["eval", "guardrails", "prompt", "testing", "reliability"],
        "hook": "Companies pay for people who can test and make models reliable. No building, just breaking well.",
        "stack": ["LangSmith, Promptfoo", "Guardrails AI"],
        "projects": ["Raga Finder Eval Suite", "Hallucination Tests"],
        "demand": "🔥 Secret Goldmine", "money": "$100k+ roles", "learn": "2w", "badge": "UNDER-RATED"
    },
    "multimodal": {
        "title": "6. Multimodal",
        "variants": ["multimodal", "image", "audio", "voice", "whisper", "diffusion", "mood"],
        "hook": "Image, audio, voice: raga → mood → image, or voice clone for meditation. Whisper + SD free.",
        "stack": ["Whisper, Stable Diffusion", "ElevenLabs"],
        "projects": ["Raga → AI Painting", "Voice Meditation"],
        "demand": "🔥 Growing", "money": "Creative premium", "learn": "3-4w", "badge": ""
    },
    "traditional ml": {
        "title": "7. Traditional ML / Data",
        "variants": ["traditional ml", "ml", "data", "forecasting", "classification", "analytics"],
        "hook": "Forecasting, classification, analytics — still 70% of real jobs, less competition than agents.",
        "stack": ["Pandas, Scikit-Learn, XGBoost", "SQL"],
        "projects": ["Concert Forecasting", "Raga Analytics"],
        "demand": "📊 70% jobs", "money": "Stable jobs", "learn": "4-6w", "badge": "STABLE"
    },
    "agentic ai": {
        "title": "8. Agentic AI (PLUS - The Future)",
        "variants": ["agentic", "agents", "agentic ai", "autonomous", "langgraph", "autogen", "crewai", "mcp"],
        "hook": "Autonomous agents that use tools, plan, remember — built ON TOP of lanes 1-7. Don't start here.",
        "stack": ["LangGraph", "AutoGen, CrewAI", "OpenAI Agents SDK, MCP, Tool Use"],
        "projects": ["Raga Research Agent (searches + writes)", "Concert Booking Agent", "Multi-Agent Bandish Studio"],
        "demand": "🚀 Future / Hype", "money": "High but needs Lanes 1-7", "learn": "6w+", "badge": "DO LAST"
    }
}

def find_lane(q):
    q=q.lower()
    for v in LANES.values():
        if any(vv in q for vv in v["variants"]): return v
    return None

st.sidebar.header("Jump to Lane")
for k,v in LANES.items():
    if st.sidebar.button(v["title"], key=k):
        st.session_state['active']=k

q = st.text_input("🔍 Search", placeholder="type rag, agent, automate, voice, lora...")
active = None
if q:
    active=find_lane(q)
    if active: st.success(f"Found → {active['title']}")
if 'active' in st.session_state:
    active=LANES[st.session_state['active']]

lanes = [active] if active else list(LANES.values())

for lane in lanes:
    with st.container(border=True):
        c1,c2,c3 = st.columns([3,1,1])
        with c1:
            st.subheader(f"{lane['title']}  {lane['badge']}")
            st.write(f"**{lane['hook']}**")
            st.caption(f"Stack: {', '.join(lane['stack'])} | Projects: {', '.join(lane['projects'])}")
        with c2:
            st.metric("Demand", lane["demand"])
            st.metric("Learn", lane["learn"])
        with c3:
            st.metric("Money", lane["money"])
            if lane["title"].startswith("8"):
                st.warning("Build after Lanes 1-5")

st.divider()
st.info("Gaurang's Path: 1 (Done ✅ Raga Finder) → 2 + 4 for $$ → 5 for reliability → 8 Agentic AI. You already nailed the hardest part!")
