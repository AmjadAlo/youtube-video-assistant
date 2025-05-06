# ------------------------------------------------------------------------
# ui: Streamlit page configuration
# ------------------------------------------------------------------------
import streamlit as st
st.set_page_config(page_title="Video QA Assistant", layout="wide")  # Wide layout for better spacing



# ------------------------------------------------------------------------
# config: Environment setup for keys and compatibility
# ------------------------------------------------------------------------
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Fix for Windows filesystem issues with Streamlit

# Use env variable for OpenAI key — DO NOT hardcode in production
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Disable unnecessary Torch profiling for faster cold starts
import torch
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)



# ------------------------------------------------------------------------
# import: Core modules and custom tools
# ------------------------------------------------------------------------
from streamlit.components.v1 import html  # For embedding raw HTML like YouTube player
from keyword_explorer import keyword_explorer  # Visual keyword summary
from picone import main_workflow  # Audio download + transcript + vector storage
import speech_recognition as sr  # For microphone-based input

# Load LangChain QA tools
from chat_with_video_voice import load_vectorstore, build_qa_chain

# LangSmith setup for tracing (optional but good for debugging)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<LANGCHAIN_API_KEY>"  # Replace with secure variable
os.environ["LANGCHAIN_PROJECT"] = "pr-grumpy-simple-26"

# Summarization and PDF/email export tools
from summary_and_email import (
    load_transcript,
    summarize_transcript,
    fetch_related_image,
    generate_pdf,
    send_email_with_pdf
)



# ------------------------------------------------------------------------
# ui: Page header + GitHub contact button
# ------------------------------------------------------------------------
col1, col2 = st.columns([6, 1])
with col1:
    st.title("🎬 Multimodal YouTube Video QA")  # Main title

with col2:
    # Custom HTML button to open GitHub profile
    github_url = "https://github.com/AmjadAlo"
    st.markdown(
        f"""
        <a href="{github_url}" target="_blank">
            <button style='margin-top: 0.7rem; padding: 0.5rem 1rem; font-size: 14px;
            font-weight: 500; background-color: white; color: #333; border: 1px solid #ccc;
            border-radius: 6px; cursor: pointer; transition: background-color 0.3s ease;'
            onmouseover="this.style.backgroundColor='#f5f5f5'" 
            onmouseout="this.style.backgroundColor='white'">
                💬 Contact Developer
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )



# ------------------------------------------------------------------------
# input: Video URL input field
# ------------------------------------------------------------------------
video_url = st.text_input("🔗 Paste the YouTube video URL here:", key="video_url_input")



# ------------------------------------------------------------------------
# state: Initialize session state variables for flow control
# ------------------------------------------------------------------------
if "processed" not in st.session_state:
    st.session_state.processed = False  # Has video been processed?
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # User Q&A history
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False  # Was quiz submitted?
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0  # Score counter
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = []  # Selected answers



# ------------------------------------------------------------------------
# helper: Embed YouTube video on screen
# ------------------------------------------------------------------------
def show_youtube_embed(url):
    if "youtube.com/watch?v=" in url or "youtu.be/" in url:
        video_id = url.split("v=")[-1] if "v=" in url else url.split("/")[-1]
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        html(f'<iframe width="560" height="315" src="{embed_url}" frameborder="0" allowfullscreen></iframe>', height=335)



# ------------------------------------------------------------------------
# helper: Voice recognition using Google STT
# ------------------------------------------------------------------------
def listen_to_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak your question.")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        st.success(f"🗣️ You said: {query}")
        return query
    except sr.UnknownValueError:
        st.warning("⚠️ Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"❌ Request error from Google Speech Recognition: {e}")
    return ""



# ------------------------------------------------------------------------
# ui: Left = video display | Right = trigger processing
# ------------------------------------------------------------------------
col1, col2 = st.columns([1.5, 1])
with col1:
    if video_url:
        show_youtube_embed(video_url)

with col2:
    if st.button("▶️ Start Processing"):
        if not video_url:
            st.warning("⚠️ Please enter a URL.")
        else:
            with st.spinner("⏳ Processing..."):
                try:
                    result = main_workflow(video_url)  # download + transcribe + embed
                    st.session_state.result = result
                    st.session_state.processed = True
                    st.success("✅ Video processed!")
                    st.info(result)
                except Exception as e:
                    st.session_state.processed = False
                    st.error(f"❌ Error during processing: {e}")



# ------------------------------------------------------------------------
# main: Run post-processing interface if video was processed
# ------------------------------------------------------------------------
if st.session_state.get("processed", False):
    st.markdown("---")
    st.markdown("### 🤖 Ask a Question About This Video")

    try:
        # Load QA system
        from chat_with_video import load_vectorstore, build_qa_chain
        with open("current_namespace.txt", "r") as f:
            namespace = f.read().strip()
        st.success(f"📂 Namespace loaded: {namespace}")

        vectordb = load_vectorstore(namespace)
        qa_chain = build_qa_chain(vectordb)

        # Text input for QA
        question = st.text_input("Type your question here:", key="user_question_input")
        if st.button("💬 Ask Question", key="submit_question"):
            with st.spinner("🧠 Thinking..."):
                result = qa_chain({"query": question})
                st.session_state.chat_history.append((question, result))
                st.markdown(f"**Answer:** {result}")

        # Show full chat history
        if st.session_state.chat_history:
            st.markdown("### 🕘 Chat History")
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")

        # Voice QA
        if st.button("🎤 Ask with Voice", key="ask_with_voice"):
            try:
                voice_question = listen_to_voice()
                if voice_question:
                    with st.spinner("🧠 Thinking..."):
                        result = qa_chain({"query": voice_question})
                        st.session_state.chat_history.append((voice_question, result))
                        st.markdown(f"**Answer:** {result}")
            except Exception as e:
                st.error(f"❌ Voice QA failed: {e}")



        # ------------------------------------------------------------------------
        # feat: Generate multiple-choice quiz from transcript
        # ------------------------------------------------------------------------
        if st.button("🧠 Generate Quiz from Video", key="generate_quiz_button"):
            with st.spinner("Generating quiz..."):
                try:
                    from quiz_generator import load_transcript, generate_quiz_questions, parse_questions
                    transcript_text = load_transcript()
                    raw_quiz = generate_quiz_questions(transcript_text, num_questions=5)
                    questions = parse_questions(raw_quiz)

                    if not questions:
                        st.error("❌ Failed to generate quiz questions.")
                    else:
                        st.session_state.generated_questions = questions
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_answers = [""] * len(questions)
                        st.success(f"✅ Quiz generated with {len(questions)} questions!")
                except Exception as e:
                    st.error(f"❌ Quiz generation failed: {e}")

        # Quiz UI: Show one radio button per question
        if "generated_questions" in st.session_state and not st.session_state.quiz_submitted:
            st.markdown("### 🎯 Take the Quiz")
            for i, (q, opts, correct) in enumerate(st.session_state.generated_questions):
                st.markdown(f"**Q{i+1}: {q}**")
                user_answer = st.radio(f"Choose your answer:", opts, key=f"quiz_radio_q{i}")
                st.session_state.quiz_answers[i] = user_answer[0] if user_answer else ""

            if st.button("✅ Submit Answers", key="submit_quiz_button"):
                score = 0
                for i, (_, _, correct) in enumerate(st.session_state.generated_questions):
                    if st.session_state.quiz_answers[i] == correct:
                        score += 1
                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True

        # Quiz results
        if st.session_state.quiz_submitted:
            total = len(st.session_state.generated_questions)
            st.markdown(f"### 🏁 Final Score: {st.session_state.quiz_score} out of {total}")
            for i, (q, opts, correct) in enumerate(st.session_state.generated_questions):
                st.markdown(f"**Q{i+1}: {q}**")
                for o in opts:
                    mark = "✅" if o.startswith(correct) else "❌" if o.startswith(st.session_state.quiz_answers[i]) else ""
                    st.markdown(f"- {o} {mark}")
                st.markdown(f"🟢 Correct Answer: {correct}")
                st.markdown("---")



        # ------------------------------------------------------------------------
        # feat: Generate summary and optionally email it as a PDF
        # ------------------------------------------------------------------------
        st.markdown("### 📝 Generate Summary and Send to Emails")

        if st.button("🧾 Generate Summary", key="generate_summary_button"):
            try:
                transcript, namespace = load_transcript()
                summary = summarize_transcript(transcript)
                st.session_state.summary_text = summary
                st.session_state.video_title = namespace.replace("_", " ").title()
                st.success("✅ Summary generated:")
                st.text_area("📄 Summary:", summary, height=250)
            except Exception as e:
                st.error(f"❌ Failed to summarize: {e}")

        if "summary_text" in st.session_state:
            summary_emails = st.text_input("📬 Emails (comma-separated):")
            sender_email = st.text_input("📤 Your Email (Gmail):")
            sender_password = st.text_input("🔑 Your App Password:", type="password")

            if st.button("✉️ Generate PDF & Send", key="send_summary_button"):
                try:
                    video_title = st.session_state.get("video_title", "Video Summary")
                    image = fetch_related_image(video_title)
                    pdf_path = generate_pdf(st.session_state.summary_text, video_title, image)
                    to_emails = [e.strip() for e in summary_emails.split(",")]

                    send_email_with_pdf(pdf_path, to_emails, sender_email, sender_password)
                    st.success("📨 Email sent with attached PDF!")
                except Exception as e:
                    st.error(f"❌ Failed to generate/send PDF: {e}")



        # ------------------------------------------------------------------------
        # feat: Show extracted keywords from transcript
        # ------------------------------------------------------------------------
        if st.button("🔑 Show Keywords"):
            keyword_explorer()

    except Exception as e:
        st.error(f"❌ QA system failed: {e}")

# If not yet processed
else:
    st.info("📌 Please process a video to start asking questions.")
