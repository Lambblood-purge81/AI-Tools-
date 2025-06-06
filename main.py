import streamlit as st
import sqlite3
import datetime
import threading
import time
import tempfile
import os
import atexit
import random
import numpy as np
from openai_utils import generate_gpt_response_with_history
from speech_utils import record_audio, transcribe_audio, AudioRecorder
from feedback_utils import analyze_response
from screen_utils import ScreenShareManager
from audio_utils import play_audio

# Constants
MAX_HISTORY_LENGTH = 20
AUDIO_TIMEOUT = 30  # seconds
FEATURE_CONFIG = {
    "Mock Interview Assistant": {
        "system_prompt": "You are a professional interview coach. Ask insightful follow-up questions and provide constructive feedback. Ask one question at a time and wait for the response.",
        "greeting": "Hello! I'll be your mock interview coach today. Let's begin with your introduction. Tell me about yourself."
    },
    "Interview Cracker": {
        "system_prompt": "You are an expert at helping candidates crack tough interviews. Analyze questions and provide strategic answers. Ask challenging questions one at a time.",
        "greeting": "Welcome to Interview Cracker! I'll help you practice tough interview questions. Let's start with a common question: What is your greatest strength?"
    }
}

# New feature configurations
FEATURE_CONFIG.update({
    "Cover Letter Generator": {
        "system_prompt": "You are a professional resume writer. Help create tailored cover letters based on job descriptions and resumes.",
        "greeting": "Let's create a compelling cover letter. Please provide the job description and your resume details."
    },
    "Interview Q&A Generator": {
        "system_prompt": "Generate common interview questions and model answers for specific job roles.",
        "greeting": "I'll help you prepare for interviews by generating likely questions and answers."
    },
    "Grammar & Tone Enhancer": {
        "system_prompt": "You are an expert editor. Improve grammar, tone, and clarity of written content while preserving the original meaning.",
        "greeting": "Paste your text below and I'll enhance its grammar and tone."
    },
    "Speech Speed Analyzer": {
        "system_prompt": "Analyze speech patterns including speed, pauses, and filler words from audio transcripts.",
        "greeting": "Record your speech and I'll analyze your speaking patterns."
    },
    "Job Match Finder": {
        "system_prompt": "Match candidate skills and preferences with suitable job opportunities.",
        "greeting": "Let's find jobs that match your skills and preferences."
    },
    "LinkedIn Summary Generator": {
        "system_prompt": "Create professional LinkedIn profile summaries based on career history and goals.",
        "greeting": "I'll help you craft an impressive LinkedIn summary."
    },
    "Career Advice Bot": {
        "system_prompt": "Provide expert career guidance and development advice.",
        "greeting": "I'm here to help with your career questions and decisions."
    },
    "AI Mentor Bot": {
        "system_prompt": "Act as a personalized career mentor providing tailored advice and resources.",
        "greeting": "Hello! I'll be your AI career mentor. How can I help you today?"
    }
})

# Page config
st.set_page_config(
    page_title="AI Career Coach Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🧠 AI Career Coach Pro")

# Initialize all session state variables
def init_session_state():
    defaults = {
        "history": [],
        "interview_active": False,
        "screen_shared": False,
        "audio_recorder": None,
        "screen_manager": None,
        "last_transcript": "",
        "conversation_state": "waiting",
        "active_feature": None,
        "temp_files": [],
        "behavioral_questions": [
            "Tell me about a time you faced a conflict at work",
            "Describe a situation where you showed leadership",
            "Give an example of how you handled failure"
        ],
        "current_question": None,
        "user_answer": "",
        "current_challenge": None,
        "technical_questions": [],
        "resume_text": "",
        "company_research_query": "",
        "job_description": "",
        "cover_letter_input": "",
        "grammar_text": "",
        "speech_analysis_data": None,
        "job_match_data": None,
        "linkedin_summary_input": "",
        "career_advice_query": "",
        "mentor_session_active": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Database setup (unchanged)
def init_db():
    try:
        conn = sqlite3.connect("chat_history.db", check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                feature TEXT
            )
        """)
        
        conn.commit()
        return conn, cursor
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                feature TEXT
            )
        """)
        conn.commit()
        return conn, cursor

conn, cursor = init_db()

def save_message(role, content, feature=None):
    try:
        content_str = str(content)
        cursor.execute("""
            INSERT INTO history (timestamp, role, content, feature) 
            VALUES (?, ?, ?, ?)
        """, (datetime.datetime.now().isoformat(), role, content_str, feature))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Failed to save message: {e}")

def get_history(feature=None, limit=MAX_HISTORY_LENGTH):
    try:
        if feature:
            cursor.execute("""
                SELECT role, content FROM history 
                WHERE feature = ? 
                ORDER BY id DESC LIMIT ?
            """, (feature, limit))
        else:
            cursor.execute("""
                SELECT role, content FROM history 
                ORDER BY id DESC LIMIT ?
            """, (limit,))
        return cursor.fetchall()[::-1]
    except sqlite3.Error as e:
        st.error(f"Failed to load history: {e}")
        return []

# Utility functions (unchanged except for new additions)
def cleanup_temp_files():
    if "temp_files" not in st.session_state:
        st.session_state.temp_files = []
        return
    
    for file in st.session_state.temp_files[:]:
        try:
            if os.path.exists(file):
                os.unlink(file)
            st.session_state.temp_files.remove(file)
        except Exception as e:
            st.error(f"Error cleaning up temp file {file}: {e}")

def safe_play_audio(text, audio_file):
    try:
        play_audio(text, audio_file)
        st.session_state.temp_files.append(audio_file)
        st.audio(audio_file, format='audio/wav')
    except Exception as e:
        st.error(f"Error generating speech: {e}")

def process_interview_question(question, feature):
    if feature not in FEATURE_CONFIG:
        st.error("Invalid feature configuration")
        return
    
    st.session_state.conversation_state = "responding"
    response = ""
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        messages = [{"role": "user", "content": question}]
        
        if feature in FEATURE_CONFIG:
            messages.insert(0, {
                "role": "system", 
                "content": FEATURE_CONFIG[feature]["system_prompt"]
            })
        
        try:
            full_response = ""
            for chunk in generate_gpt_response_with_history(messages):
                if isinstance(chunk, str):
                    full_response += chunk
                else:
                    full_response += str(chunk)
            
            placeholder.markdown(full_response)
            response = full_response
        except Exception as e:
            st.error(f"Error generating response: {e}")
            response = "I encountered an error processing your request. Please try again."
            placeholder.markdown(response)
    
    st.session_state.history.append({"role": "assistant", "content": response})
    save_message("assistant", response, feature)
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            audio_file = tmpfile.name
            safe_play_audio(response, audio_file)
    except Exception as e:
        st.error(f"Error creating audio file: {e}")
    
    st.session_state.conversation_state = "waiting"
    st.rerun()

# Enhanced interview functions for real-time interaction
def start_interview(feature):
    if feature not in FEATURE_CONFIG:
        st.error("Invalid feature selected")
        return
    
    try:
        st.session_state.interview_active = True
        st.session_state.active_feature = feature
        st.session_state.audio_recorder = AudioRecorder()
        st.session_state.audio_recorder.start()
        
        threading.Thread(
            target=interview_thread, 
            args=(feature,),
            daemon=True
        ).start()
        
        greeting = FEATURE_CONFIG[feature].get("greeting", "Let's begin the interview.")
        
        st.session_state.history.append({"role": "assistant", "content": greeting})
        save_message("assistant", greeting, feature)
        
        with st.chat_message("assistant"):
            st.markdown(greeting)
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                audio_file = tmpfile.name
                safe_play_audio(greeting, audio_file)
        except Exception as e:
            st.error(f"Error generating greeting: {e}")
        
        st.rerun()
    except Exception as e:
        st.error(f"Failed to start interview: {e}")
        stop_interview()

def interview_thread(feature):
    start_time = time.time()
    
    while (st.session_state.interview_active and 
           st.session_state.active_feature == feature and
           time.time() - start_time < AUDIO_TIMEOUT * 10):  # Increased timeout for longer sessions
        
        try:
            if st.session_state.conversation_state == "waiting":
                st.session_state.conversation_state = "listening"
                
                audio_data = st.session_state.audio_recorder.get_audio_data()
                if audio_data is not None and len(audio_data) > 0:
                    transcript = transcribe_audio(audio_data)
                    
                    if transcript and transcript != st.session_state.last_transcript:
                        st.session_state.last_transcript = transcript
                        st.session_state.history.append({"role": "user", "content": transcript})
                        save_message("user", transcript, feature)
                        
                        # Real-time analysis of speech patterns
                        if feature == "Speech Speed Analyzer":
                            analyze_speech_patterns(transcript)
                        
                        st.session_state.conversation_state = "processing"
                        st.rerun()
            
            time.sleep(0.5)
        except Exception as e:
            st.error(f"Interview thread error: {e}")
            break

# New feature functions
def generate_cover_letter():
    if not st.session_state.job_description or not st.session_state.cover_letter_input:
        st.warning("Please provide both job description and your background info")
        return
    
    try:
        messages = [
            {"role": "system", "content": FEATURE_CONFIG["Cover Letter Generator"]["system_prompt"]},
            {"role": "user", "content": f"Job Description:\n{st.session_state.job_description}\n\nMy Background:\n{st.session_state.cover_letter_input}"}
        ]
        
        full_response = ""
        for chunk in generate_gpt_response_with_history(messages):
            if isinstance(chunk, str):
                full_response += chunk
        
        st.session_state.history.append({"role": "assistant", "content": full_response})
        save_message("assistant", full_response, "Cover Letter Generator")
        
        with st.chat_message("assistant"):
            st.markdown(full_response)
        
        return full_response
    except Exception as e:
        st.error(f"Error generating cover letter: {e}")
        return None

def analyze_speech_patterns(transcript):
    try:
        messages = [
            {"role": "system", "content": FEATURE_CONFIG["Speech Speed Analyzer"]["system_prompt"]},
            {"role": "user", "content": f"Analyze this speech:\n{transcript}"}
        ]
        
        analysis = ""
        for chunk in generate_gpt_response_with_history(messages):
            if isinstance(chunk, str):
                analysis += chunk
        
        st.session_state.speech_analysis_data = analysis
        save_message("assistant", analysis, "Speech Speed Analyzer")
        
        with st.chat_message("assistant"):
            st.markdown(analysis)
        
        return analysis
    except Exception as e:
        st.error(f"Error analyzing speech: {e}")
        return None

def enhance_grammar():
    if not st.session_state.grammar_text:
        st.warning("Please enter text to enhance")
        return
    
    try:
        messages = [
            {"role": "system", "content": FEATURE_CONFIG["Grammar & Tone Enhancer"]["system_prompt"]},
            {"role": "user", "content": f"Improve this text:\n{st.session_state.grammar_text}"}
        ]
        
        enhanced_text = ""
        for chunk in generate_gpt_response_with_history(messages):
            if isinstance(chunk, str):
                enhanced_text += chunk
        
        st.session_state.history.append({"role": "assistant", "content": enhanced_text})
        save_message("assistant", enhanced_text, "Grammar & Tone Enhancer")
        
        with st.chat_message("assistant"):
            st.markdown("**Enhanced Version:**")
            st.markdown(enhanced_text)
            st.markdown("**Changes Made:**")
            
            # Get explanation of changes
            explain_messages = [
                {"role": "system", "content": "Explain the grammar and tone improvements you made"},
                {"role": "assistant", "content": enhanced_text},
                {"role": "user", "content": st.session_state.grammar_text}
            ]
            
            explanation = ""
            for chunk in generate_gpt_response_with_history(explain_messages):
                if isinstance(chunk, str):
                    explanation += chunk
            
            st.markdown(explanation)
            save_message("assistant", explanation, "Grammar & Tone Enhancer")
        
        return enhanced_text
    except Exception as e:
        st.error(f"Error enhancing text: {e}")
        return None

# Sidebar with all features
app_mode = st.sidebar.selectbox("Select Feature", [
    "Mock Interview Assistant",
    "Interview Cracker",
    "Cover Letter Generator",
    "Interview Q&A Generator",
    "Grammar & Tone Enhancer",
    "Speech Speed Analyzer",
    "Job Match Finder",
    "LinkedIn Summary Generator",
    "Career Advice Bot",
    "AI Mentor Bot"
])

# Main content layout
col1, col2 = st.columns([2, 1])

with col1:
    st.session_state.active_feature = app_mode
    
    # Interview Features (unchanged)
    if app_mode in ["Mock Interview Assistant", "Interview Cracker"]:
        st.header(f"🎥 {app_mode}")
        st.markdown("""
        **Realistic Interview Experience**  
        - Practice with AI that listens and responds naturally
        - Get real-time feedback on your answers
        - Simulate actual interview conditions
        """)
        
        if not st.session_state.screen_shared:
            if st.button("🖥️ Share Your Screen"):
                try:
                    st.session_state.screen_manager = ScreenShareManager()
                    st.session_state.screen_shared = True
                    st.success("Screen sharing initialized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize screen sharing: {e}")
        else:
            st.success("✅ Screen sharing is active")
            if st.button("🛑 Stop Screen Sharing"):
                try:
                    st.session_state.screen_manager.stop()
                    st.session_state.screen_shared = False
                    st.session_state.screen_manager = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop screen sharing: {e}")
        
        if not st.session_state.interview_active:
            if st.button(f"🎤 Start {app_mode.split()[0]} Session"):
                start_interview(app_mode)
        else:
            if st.session_state.active_feature == app_mode:
                if st.button("⏹️ End Session"):
                    stop_interview()
                
                states = {
                    "waiting": "🟢 Ready for your response",
                    "listening": "🎤 Listening...",
                    "processing": "🤔 Processing your answer...",
                    "responding": "💬 AI is responding"
                }
                current_state = st.session_state.conversation_state
                st.info(states.get(current_state, "🟠 Unknown state"))
                
                if current_state == "processing":
                    process_interview_question(
                        st.session_state.last_transcript, 
                        app_mode
                    )
    
    # Cover Letter Generator Feature
    elif app_mode == "Cover Letter Generator":
        st.header("📝 Cover Letter Generator")
        st.markdown("Create a tailored cover letter based on job description and your background")
        
        st.session_state.job_description = st.text_area(
            "Paste the job description:",
            height=150,
            value=st.session_state.job_description
        )
        
        st.session_state.cover_letter_input = st.text_area(
            "Provide your background information (skills, experience, achievements):",
            height=200,
            value=st.session_state.cover_letter_input
        )
        
        if st.button("Generate Cover Letter"):
            with st.spinner("Creating your customized cover letter..."):
                cover_letter = generate_cover_letter()
                if cover_letter:
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmpfile:
                            # Save as downloadable file
                            import docx
                            doc = docx.Document()
                            doc.add_paragraph(cover_letter)
                            doc.save(tmpfile.name)
                            st.session_state.temp_files.append(tmpfile.name)
                            
                            with open(tmpfile.name, "rb") as f:
                                st.download_button(
                                    label="Download Cover Letter",
                                    data=f,
                                    file_name="tailored_cover_letter.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                    except Exception as e:
                        st.error(f"Error creating document: {e}")

    # Interview Q&A Generator Feature
    elif app_mode == "Interview Q&A Generator":
        st.header("❓ Interview Q&A Generator")
        job_title = st.text_input("Job Title you're applying for:")
        experience_level = st.selectbox("Your Experience Level", 
                                      ["Entry Level", "Mid Level", "Senior Level"])
        industry = st.text_input("Industry (optional):")
        
        if st.button("Generate Likely Questions"):
            if not job_title:
                st.warning("Please enter a job title")
            else:
                with st.spinner("Generating likely interview questions and answers..."):
                    try:
                        prompt = f"Generate 10 likely interview questions for a {experience_level} {job_title} position"
                        if industry:
                            prompt += f" in the {industry} industry"
                        prompt += ". For each question, provide a model answer."
                        
                        messages = [
                            {"role": "system", "content": FEATURE_CONFIG["Interview Q&A Generator"]["system_prompt"]},
                            {"role": "user", "content": prompt}
                        ]
                        
                        qa_content = ""
                        for chunk in generate_gpt_response_with_history(messages):
                            if isinstance(chunk, str):
                                qa_content += chunk
                        
                        st.session_state.history.append({"role": "assistant", "content": qa_content})
                        save_message("assistant", qa_content, "Interview Q&A Generator")
                        
                        st.markdown(qa_content)
                        
                        # Add practice section
                        st.subheader("Practice Your Answers")
                        st.markdown("Select a question to practice answering:")
                        
                        questions = [q for q in qa_content.split("\n") if q.strip() and (q.startswith("1.") or "?" in q)]
                        selected_q = st.selectbox("Select a question:", questions)
                        
                        if st.button("Record My Answer"):
                            audio_file = record_audio()
                            if isinstance(audio_file, np.ndarray):
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                                    import soundfile as sf
                                    sf.write(tmpfile.name, audio_file, 16000)
                                    audio_file = tmpfile.name
                                    st.session_state.temp_files.append(audio_file)
                            
                            transcript = transcribe_audio(audio_file)
                            st.session_state.user_answer = transcript
                            
                            feedback_prompt = f"Evaluate this answer to '{selected_q}':\n{transcript}\n\nProvide specific feedback on content, structure, and delivery."
                            
                            feedback = ""
                            for chunk in generate_gpt_response_with_history([{"role": "user", "content": feedback_prompt}]):
                                if isinstance(chunk, str):
                                    feedback += chunk
                            
                            st.write("**Feedback:**", feedback)
                            save_message("user", transcript, "Interview Q&A Generator")
                            save_message("assistant", feedback, "Interview Q&A Generator")
                    except Exception as e:
                        st.error(f"Error generating Q&A: {e}")

    # Grammar & Tone Enhancer Feature
    elif app_mode == "Grammar & Tone Enhancer":
        st.header("✍️ Grammar & Tone Enhancer")
        st.markdown("Improve your writing with advanced grammar and tone suggestions")
        
        st.session_state.grammar_text = st.text_area(
            "Enter your text to enhance:",
            height=300,
            value=st.session_state.grammar_text,
            key="grammar_input"
        )
        
        tone = st.selectbox(
            "Desired Tone (optional):",
            ["Professional", "Casual", "Academic", "Persuasive", "Friendly", "Formal"]
        )
        
        if st.button("Enhance Text"):
            if not st.session_state.grammar_text:
                st.warning("Please enter some text to enhance")
            else:
                with st.spinner("Analyzing and improving your text..."):
                    if tone != "Professional":
                        st.session_state.grammar_text += f"\n\nPlease make the tone more {tone.lower()}."
                    enhanced_text = enhance_grammar()

    # Speech Speed Analyzer Feature
    elif app_mode == "Speech Speed Analyzer":
        st.header("🎤 Speech Speed Analyzer")
        st.markdown("Analyze your speech patterns for interviews and presentations")
        
        if not st.session_state.interview_active:
            if st.button("Start Speech Analysis Session"):
                start_interview(app_mode)
        else:
            if st.session_state.active_feature == app_mode:
                if st.button("Stop Analysis"):
                    stop_interview()
                
                st.info("🎤 Recording your speech... Speak naturally")
                
                if st.session_state.speech_analysis_data:
                    st.subheader("Analysis Results")
                    st.markdown(st.session_state.speech_analysis_data)
                    
                    # Visualization of speech metrics
                    st.subheader("Speech Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    # Sample metrics (in a real app, these would come from actual analysis)
                    col1.metric("Words per Minute", "145", "-5 from ideal")
                    col2.metric("Pause Frequency", "3.2/sec", "High")
                    col3.metric("Filler Words", "12%", "8% over target")
                    
                    st.progress(65, text="Overall Speech Score")

    # Job Match Finder Feature
    elif app_mode == "Job Match Finder":
        st.header("🔍 Job Match Finder")
        st.markdown("Find jobs that match your skills and preferences")
        
        skills = st.text_area("Your Skills (comma separated):")
        experience = st.text_input("Years of Experience:")
        location = st.text_input("Preferred Location (optional):")
        salary_exp = st.text_input("Salary Expectations (optional):")
        
        if st.button("Find Matching Jobs"):
            if not skills or not experience:
                st.warning("Please provide at least skills and experience")
            else:
                with st.spinner("Searching for matching jobs..."):
                    try:
                        prompt = f"Find job matches for someone with these skills: {skills} and {experience} years experience"
                        if location:
                            prompt += f" in {location}"
                        if salary_exp:
                            prompt += f" with salary expectations around {salary_exp}"
                        
                        messages = [
                            {"role": "system", "content": FEATURE_CONFIG["Job Match Finder"]["system_prompt"]},
                            {"role": "user", "content": prompt}
                        ]
                        
                        matches = ""
                        for chunk in generate_gpt_response_with_history(messages):
                            if isinstance(chunk, str):
                                matches += chunk
                        
                        st.session_state.job_match_data = matches
                        st.session_state.history.append({"role": "assistant", "content": matches})
                        save_message("assistant", matches, "Job Match Finder")
                        
                        st.markdown(matches)
                    except Exception as e:
                        st.error(f"Error finding job matches: {e}")

    # LinkedIn Summary Generator Feature
    elif app_mode == "LinkedIn Summary Generator":
        st.header("🔗 LinkedIn Summary Generator")
        st.markdown("Create a professional LinkedIn profile summary")
        
        st.session_state.linkedin_summary_input = st.text_area(
            "Provide your career information (experience, skills, achievements, goals):",
            height=250,
            value=st.session_state.linkedin_summary_input
        )
        
        tone = st.selectbox(
            "Summary Tone:",
            ["Professional", "Creative", "Technical", "Executive", "Entrepreneurial"]
        )
        
        if st.button("Generate Summary"):
            if not st.session_state.linkedin_summary_input:
                st.warning("Please provide your career information")
            else:
                with st.spinner("Creating your professional summary..."):
                    try:
                        prompt = f"Create a {tone.lower()} LinkedIn summary based on this information:\n{st.session_state.linkedin_summary_input}"
                        
                        messages = [
                            {"role": "system", "content": FEATURE_CONFIG["LinkedIn Summary Generator"]["system_prompt"]},
                            {"role": "user", "content": prompt}
                        ]
                        
                        summary = ""
                        for chunk in generate_gpt_response_with_history(messages):
                            if isinstance(chunk, str):
                                summary += chunk
                        
                        st.session_state.history.append({"role": "assistant", "content": summary})
                        save_message("assistant", summary, "LinkedIn Summary Generator")
                        
                        st.subheader("Your LinkedIn Summary")
                        st.markdown(summary)
                        
                        # Add optimization tips
                        st.subheader("Optimization Tips")
                        tips_prompt = "Provide 3-5 tips to optimize this LinkedIn profile for maximum visibility to recruiters"
                        
                        tips = ""
                        for chunk in generate_gpt_response_with_history([
                            {"role": "assistant", "content": summary},
                            {"role": "user", "content": tips_prompt}
                        ]):
                            if isinstance(chunk, str):
                                tips += chunk
                        
                        st.markdown(tips)
                        save_message("assistant", tips, "LinkedIn Summary Generator")
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

    # Career Advice Bot Feature
    elif app_mode == "Career Advice Bot":
        st.header("💼 Career Advice Bot")
        st.markdown("Get personalized career guidance and advice")
        
        st.session_state.career_advice_query = st.text_area(
            "What career advice are you looking for?",
            height=150,
            value=st.session_state.career_advice_query,
            key="career_advice_input"
        )
        
        if st.button("Get Advice"):
            if not st.session_state.career_advice_query:
                st.warning("Please enter your career question")
            else:
                with st.spinner("Analyzing your career question..."):
                    try:
                        messages = [
                            {"role": "system", "content": FEATURE_CONFIG["Career Advice Bot"]["system_prompt"]},
                            {"role": "user", "content": st.session_state.career_advice_query}
                        ]
                        
                        advice = ""
                        for chunk in generate_gpt_response_with_history(messages):
                            if isinstance(chunk, str):
                                advice += chunk
                        
                        st.session_state.history.append({"role": "assistant", "content": advice})
                        save_message("assistant", advice, "Career Advice Bot")
                        
                        st.markdown(advice)
                    except Exception as e:
                        st.error(f"Error getting career advice: {e}")

    # AI Mentor Bot Feature
    elif app_mode == "AI Mentor Bot":
        st.header("🤖 AI Mentor Bot")
        st.markdown("Your personalized career mentor for ongoing guidance")
        
        if not st.session_state.mentor_session_active:
            if st.button("Start Mentor Session"):
                st.session_state.mentor_session_active = True
                st.session_state.history.append({
                    "role": "assistant",
                    "content": FEATURE_CONFIG["AI Mentor Bot"]["greeting"]
                })
                save_message("assistant", FEATURE_CONFIG["AI Mentor Bot"]["greeting"], "AI Mentor Bot")
                st.rerun()
        else:
            user_input = st.text_input("What would you like to discuss with your mentor?", key="mentor_input")
            
            if user_input:
                st.session_state.history.append({"role": "user", "content": user_input})
                save_message("user", user_input, "AI Mentor Bot")
                
                with st.spinner("Your mentor is thinking..."):
                    try:
                        # Build conversation history
                        messages = [{"role": "system", "content": FEATURE_CONFIG["AI Mentor Bot"]["system_prompt"]}]
                        for msg in st.session_state.history[-5:]:  # Use recent history for context
                            messages.append({"role": msg["role"], "content": msg["content"]})
                        
                        response = ""
                        for chunk in generate_gpt_response_with_history(messages):
                            if isinstance(chunk, str):
                                response += chunk
                        
                        st.session_state.history.append({"role": "assistant", "content": response})
                        save_message("assistant", response, "AI Mentor Bot")
                        
                        st.markdown(f"**Mentor:** {response}")
                    except Exception as e:
                        st.error(f"Error in mentor session: {e}")
            
            if st.button("End Mentor Session"):
                st.session_state.mentor_session_active = False
                st.session_state.history = []
                st.rerun()

with col2:
    st.subheader("📁 Recent History")
    history = get_history(st.session_state.active_feature)
    
    if not history:
        st.info("No history yet. Start a conversation to see it here.")
    else:
        for role, msg in history:
            with st.expander(f"{role.capitalize()}: {msg[:50]}..." if len(msg) > 50 else f"{role.capitalize()}: {msg}"):
                st.write(msg)

# Cleanup handler (unchanged)
def cleanup():
    try:
        if st.session_state.get("audio_recorder"):
            st.session_state.audio_recorder.stop()
        if st.session_state.get("screen_manager"):
            st.session_state.screen_manager.stop()
        cleanup_temp_files()
    except Exception as e:
        st.error(f"Cleanup error: {e}")

atexit.register(cleanup)

# Custom CSS (unchanged)
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        transition: all 0.2s;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background-color: #45a049;
    }
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        padding: 8px;
    }
    .stSelectbox select {
        border-radius: 8px;
        padding: 8px;
    }
    .st-expander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .st-expander:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)