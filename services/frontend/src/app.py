import streamlit as st
import requests
import json
import os
import asyncio
import io
import uuid
import time
from services.frontend.src.realtime_client import RealtimeClient

from shared.src.config import settings

# Configuration
ORCHESTRATOR_URL = settings.ORCHESTRATOR_URL
AGENT_REGISTRY_URL = settings.AGENT_REGISTRY_URL

st.set_page_config(page_title="EPBA Medical Assistant", page_icon="🏥", layout="wide")
# ─── Custom CSS for polished chat styling ─────────────────────────────────────
st.markdown("""
<style>
    /* Pin chat input above bottom, wide but with margins */
    /* Chat Input: Fixed Bottom, Dynamically Aligned with Header */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        /* Match main content padding (5rem = approx 80px) */
        left: 5rem !important;
        right: 5rem !important;
        width: auto !important;
        max-width: none !important;
        background: linear-gradient(to top, #0e1117, transparent);
        z-index: 90;
        padding-bottom: 1rem !important;
        padding-top: 1rem !important;
        /* Smoothly animate position changes to match sidebar slide */
        transition: left 0.3s ease-in-out, width 0.3s ease-in-out !important;
    }

    /* Shift right when sidebar is open (300px sidebar + 5rem padding) */
    /* Note: Use ~ * because section.main might be wrapped in a div */
    section[data-testid="stSidebar"][aria-expanded="true"] ~ * [data-testid="stChatInput"] {
        left: calc(300px + 5rem) !important;
    }

    [data-testid="stChatInput"] > div {
        margin: 0 auto;
        width: 100% !important;
        max-width: none !important;
    }
    textarea[aria-label="What would you like to know?"] {
         color: white !important;
         padding-right: 10rem !important; /* Make room for Mic and spacing */
    }
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border-radius: 20px;
        background: #262730;
        border: 2px solid #000000 !important;
        color: white;
        width: 100%;
    }
    .stChatInput textarea {
        color: white !important;
        width: 100% !important;
    }
    .stChatInput ::placeholder {
        color: #b0b0b0 !important;
    }
    
    /* ── Mic button: overlay on chat input bar ─────────────────── */
    /* Target the Streamlit block that contains the audio input widget */
    [data-testid="stAudioInput"] {
        position: fixed !important;
        bottom: 20px !important;
        right: 8.5rem !important; /* Move left to create space from the send button */
        z-index: 200 !important;
        width: 40px !important;
        height: 40px !important;
        overflow: visible !important;
        background: transparent !important;
        border: none !important;
    }
    /* Hide label */
    [data-testid="stAudioInput"] > label { display: none !important; }
    /* Strip the inner container chrome */
    [data-testid="stAudioInput"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        min-height: 0 !important;
        width: 40px !important;
        height: 40px !important;
        overflow: hidden !important;
    }
    /* Hide waveform, timer — keep only the mic/stop button */
    [data-testid="stAudioInput"] [data-testid="stAudioInputWaveSurfer"] { display: none !important; }
    [data-testid="stAudioInput"] [data-testid="stAudioInputTimer"] { display: none !important; }
    /* Style the mic/stop button */
    [data-testid="stAudioInput"] button {
        background: transparent !important;
        border: none !important;
        color: #b0b0b0 !important;
        padding: 4px !important;
        margin: 0 !important;
        min-height: 0 !important;
        width: 36px !important;
        height: 36px !important;
        cursor: pointer !important;
    }
    [data-testid="stAudioInput"] button:hover {
        color: #ffffff !important;
    }
    [data-testid="stAudioInput"] button svg {
        width: 20px !important;
        height: 20px !important;
    }
    /* Recording state — pulse red */
    [data-testid="stAudioInput"] button[aria-label*="Stop"],
    [data-testid="stAudioInput"] button[aria-pressed="true"] {
        color: #ff4b4b !important;
        animation: mic-pulse 1s ease-in-out infinite;
    }
    @keyframes mic-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Increase tab font size */
    .stTabs button {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    .stTabs button p {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Compact header for Agent Directory */
    .compact-header {
        display: flex;
        align-items: baseline;
        gap: 1rem;
        padding: 0;
        margin: 0;
    }
    .compact-title {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .compact-status {
        font-size: 1rem !important;
        color: #888 !important;
        font-weight: 400 !important;
        margin: 0 !important;
    }
    
    /* Agent Card styles */
    .agent-card {
        background: #262730;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        height: 100%;
    }
    .agent-card-offline {
        border-left: 4px solid #dc3545;
    }
    .skill-tag {
        background: #1f77b4;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin-right: 4px;
        display: inline-block;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_chat, tab_agents = st.tabs(["💬 Chat", "🤖 Agent Directory"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Chat
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.title("🏥 EPBA AI Assistant")
    st.markdown("Ask questions about patient records and get comprehensive summaries.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
            
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Display chat messages from history on app rerun   
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type", "text") == "user_audio":
                st.audio(message["audio_bytes"], format="audio/wav")
                st.status("Voice Interaction Complete", state="complete", expanded=False)
            elif message.get("type", "text") == "assistant_audio":
                st.markdown(f"**Transcript**: _{message.get('transcript', '')}_")
                st.audio(message["wav_bytes"], format="audio/wav")
                if message.get("assistant_transcript"):
                    st.info(f"🔊 _{message['assistant_transcript']}_")
            else:
                st.markdown(message["content"])


    # Chat input — rendered at the bottom by Streamlit
    prompt = st.chat_input("What would you like to know?", key="chat_input_main")

    # ── MIC BUTTON (overlaid on chat input via CSS) ──────────────────────────────
    audio_val = st.audio_input("mic", key="audio_input", label_visibility="collapsed")
    
    # Logic to handle Audio Input
    if audio_val:
        # Check if we already processed this specific audio input to avoid loops
        # Streamlit re-runs script on interaction. We use a session state tracker.
        if "last_audio_id" not in st.session_state:
            st.session_state.last_audio_id = None
        
        # Simple check: assuming audio_val has a unique ID or we check bytes hash
        # st.audio_input returns a BytesIO-like object. 
        # We can just process it. If it's the same object instance, we might re-process if not careful?
        # Streamlit resets widgets on reruns often, or keeps state.
        # Ideally we process then clear, but we can't clear widget programmatically easily without key hack.
        # We'll assume the user records, we process.
        
        # We'll use a hash or ID check to prevent re-processing the same recording on unintended reruns
        # audio_val.getvalue() gives bytes. 
        audio_bytes = audio_val.getvalue()
        audio_hash = hash(audio_bytes)
        
        if st.session_state.last_audio_id != audio_hash:
            st.session_state.last_audio_id = audio_hash
            
            with st.chat_message("user"):

                st.audio(audio_val, format="audio/wav")
                status_box = st.status("Processing Voice Query...", expanded=True)
                
            # Define callbacks
            def update_status(msg):
                status_box.write(msg)
                
            def query_orchestrator(text_query):
                # Retrieve Data via Orchestrator
                try:
                    resp = requests.post(ORCHESTRATOR_URL, json={"query": text_query}, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        final = data.get("final_answer", "No answer.")
                        return final
                    return "Error contacting Orchestrator."
                except Exception as e:
                    return f"Orchestrator Error: {e}"

            # Async Process
            from realtime_client import RealtimeClient
            client = RealtimeClient()
            
            # Run async flow in sync Streamlit
            # Run async flow in sync Streamlit
            try:
                wav_bytes, transcript, orchestrator_final, assistant_transcript = asyncio.run(client.run_flow(audio_bytes, update_status, query_orchestrator, session_id=st.session_state.session_id))
                
                status_box.update(label="Voice Interaction Complete", state="complete", expanded=False)
                
                if wav_bytes:
                     # Update Chat History in Main Thread
                     st.session_state.messages.append({
                         "role": "user", 
                         "type": "user_audio",
                         "audio_bytes": audio_bytes,
                         "content": f"🎙️ {transcript}" # Used mostly for debugging or external logs
                     })
                     if orchestrator_final:
                         st.session_state.messages.append({
                             "role": "assistant", 
                             "type": "assistant_audio",
                             "transcript": transcript,
                             "wav_bytes": wav_bytes,
                             "assistant_transcript": assistant_transcript,
                             "content": orchestrator_final # optional text context
                         })
                     
                     # Play Result
                     with st.chat_message("assistant"):
                         st.markdown(f"**Transcript**: _{transcript}_")
                         st.audio(wav_bytes, format="audio/wav", autoplay=True)
                         if assistant_transcript:
                             st.info(f"🔊 _{assistant_transcript}_")
                         
                else:
                     st.error(f"Failed to generate audio response. Transcript: {transcript}")
                     
            except Exception as e:
                status_box.update(label="Voice Interaction Failed", state="error")
                st.error(f"Realtime API Error: {e}")

    # ── TEXT INPUT HANDLER ──────────────────────────────────────────────────────
    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Sidebar for Agent Trace
        with st.sidebar:
            st.header("Agent Execution Flows")
            st.info("Query Received ....")
            workflow_steps = st.container()

        # Call Orchestrator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            with st.sidebar:
                 st.info("Orchestrator: Processing...")
                 
            try:
                start_time = time.time()
                response = requests.post(
                    ORCHESTRATOR_URL, 
                    json={"query": prompt},
                    timeout=120
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    final_answer = data.get("final_answer", "No answer generated.")
                    sql_res = data.get("sql_result", "")
                    vec_res = data.get("vector_result", "")
                    timings = data.get("timings", {})
                    
                    sql_time = timings.get("sql_agent", 0)
                    vec_time = timings.get("vector_agent", 0)
                    sum_time = timings.get("summarizer", 0)
                    
                    message_placeholder.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    
                    with st.sidebar:
                        st.success(f"Execution Complete ({duration:.2f}s)")
                        st.divider()
                        
                        st.markdown("### 1. Orchestrator")
                        st.caption("Received Query & Dispatched Agents")
                        
                        st.markdown("### 2. Parallel Execution")
                        with st.expander(f"SQL Agent Status ({sql_time:.2f}s)", expanded=True):
                            if any(phrase in sql_res.lower() for phrase in ["no patients named", "no patient found", "no records found", "i don't know", "not found", "agent stopped"]):
                                 st.warning("⚠️ SQL Agent: No Records Found")
                            elif sql_res:
                                 st.success("✅ SQL Agent: Data Retrieved")
                            else:
                                 st.error("❌ SQL Agent: No Response")
                        
                        with st.expander(f"Vector Agent Status ({vec_time:.2f}s)", expanded=True):
                            if "No relevant documents" in vec_res:
                                 st.warning("⚠️ Vector Agent: No Context Found")
                            elif vec_res:
                                 st.success("✅ Vector Agent: Context Retrieved")
                            else:
                                 st.error("❌ Vector Agent: No Response")
                                 
                        st.markdown(f"### 3. Summarization ({sum_time:.2f}s)")
                        st.success("✅ Summarizer Agent: Synthesized Answer")
                        
                    
                    with st.expander("View Source Data"):
                        st.subheader("SQL Agent Result")
                        st.code(sql_res, language="text")
                        
                        st.subheader("Vector Agent Result")
                        st.code(vec_res, language="text")
                        
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Connection Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Agent Directory — Card Grid Layout
# ═══════════════════════════════════════════════════════════════════════════════

with tab_agents:
    # ── Logic to fetch agents ────────────────────────────────────────────────
    if "refresh_key" not in st.session_state:
        st.session_state.refresh_key = 0
        
    def trigger_refresh():
        st.cache_data.clear()
        st.session_state.refresh_key += 1
        
    @st.cache_data(ttl=15)
    def fetch_agents():
        try:
            resp = requests.get(f"{AGENT_REGISTRY_URL}/agents", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    registry_data = fetch_agents()
    
    # Calculate status string
    status_text = "Connecting..."
    online_count = 0
    total_count = 0
    agents = []
    
    if registry_data:
        agents = registry_data.get("agents", [])
        total_count = registry_data.get("count", 0)
        online_count = sum(1 for a in agents if a.get("status") == "online")
        status_text = f"{online_count}/{total_count} agents online &nbsp;|&nbsp; Registry: `{AGENT_REGISTRY_URL}`"
    else:
        status_text = "⚠️ Registry Offline"

    # ── Header Layout: Title + Status | Refresh Button ──────────────────────
    col_header, col_btn = st.columns([5, 1])
    
    with col_header:
        # Custom HTML for inline title + status using the CSS classes defined above
        st.markdown(f"""
        <div class="compact-header">
            <span class="compact-title">🤖 Agent Directory</span>
            <span class="compact-status">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_btn:
        st.button("🔄 Refresh", use_container_width=True, on_click=trigger_refresh, key="refresh_btn_compact")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Content ─────────────────────────────────────────────────────────────
    if registry_data is None:
        st.error(f"⚠️ Could not connect to Agent Registry at {AGENT_REGISTRY_URL}")
    else:
        if not agents:
            st.warning("No agents registered yet.")
        else:
            # ── Grid layout: 2 columns ──────────────────────────────────
            cols = st.columns(2, gap="medium")

            for idx, agent in enumerate(agents):
                card = agent.get("agent_card", {})
                status = agent.get("status", "unknown")
                base_url = agent.get("base_url", "")
                response_time = agent.get("response_time_ms")
                name = card.get("name", agent.get("name", "Unknown Agent"))
                description = card.get("description", "No description available.")
                version = card.get("version", "—")
                skills = card.get("skills", [])
                capabilities = card.get("capabilities", {})
                provider = card.get("provider", {})
                input_modes = card.get("defaultInputModes", [])
                output_modes = card.get("defaultOutputModes", [])

                status_icon = "🟢" if status == "online" else "🔴"
                status_label = "Online" if status == "online" else "Offline"
                border_class = "agent-card-online" if status == "online" else "agent-card-offline"

                with cols[idx % 2]:
                    # Card container
                    with st.container(border=True):
                        # Header
                        h_col, s_col = st.columns([3, 1])
                        with h_col:
                            st.markdown(f"#### {status_icon} {name}")
                        with s_col:
                            if status == "online":
                                st.success(status_label)
                            else:
                                st.error(status_label)

                        # Description (truncated for card view)
                        desc_short = description[:120] + "..." if len(description) > 120 else description
                        st.caption(desc_short)

                        # Metadata row
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f"**Version:** `{version}`")
                        with m2:
                            org = provider.get("organization", "—")
                            st.markdown(f"**Provider:** {org}")
                        with m3:
                            if response_time is not None:
                                st.markdown(f"**Latency:** `{response_time:.0f}ms`")
                            else:
                                st.markdown("**Latency:** —")

                        # Capabilities
                        streaming = capabilities.get("streaming", False)
                        push = capabilities.get("pushNotifications", False)
                        caps = []
                        caps.append(f"{'✅' if streaming else '❌'} Streaming")
                        caps.append(f"{'✅' if push else '❌'} Push")
                        if input_modes:
                            caps.append(f"📥 {', '.join(input_modes)}")
                        st.caption(" &nbsp;|&nbsp; ".join(caps))

                        # Skills
                        if skills:
                            for skill in skills:
                                skill_name = skill.get("name", "Unnamed")
                                skill_tags = skill.get("tags", [])
                                skill_examples = skill.get("examples", [])

                                st.markdown(f"**🛠 {skill_name}**")

                                if skill_tags:
                                    tags_html = " ".join(
                                        f'<span class="skill-tag">{t}</span>'
                                        for t in skill_tags
                                    )
                                    st.markdown(tags_html, unsafe_allow_html=True)

                                if skill_examples:
                                    with st.expander("Example queries"):
                                        for ex in skill_examples:
                                            st.markdown(f"- _{ex}_")

                        # Endpoint
                        st.caption(f"📡 `{base_url}`")

                        # Full JSON
                        with st.expander("🔍 Full Agent Card JSON"):
                            st.json(card)
