# youtube-video-assistant

# 🎬 Multimodal YouTube Video Assistant

An AI-powered Streamlit application that processes YouTube videos and enables interactive features like question answering (text & voice), quiz generation, transcript summarization, keyword extraction, and PDF email sharing.

---

## 🚀 Features

- 🔗 YouTube video processing & transcription
- 🤖 Ask questions (text or voice)
- 🧠 Auto-generate quizzes
- 🗝 Extract top keywords with Wikipedia links
- 📝 Summarize transcripts into clean overviews
- 📤 Send summaries as PDF via email

---

## ⚙️ Installation

```bash
git clone https://github.com/AmjadAlo/youtube-video-assistant.git
cd youtube-video-assistant
pip install -r requirements.txt
```

---

## 🔐 API Keys

Store these as environment variables or use directly in code (as currently configured):
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `LANGCHAIN_API_KEY`

---

## 🧭 How to Use

1. Run `streamlit_app_final.py`
2. Paste a YouTube URL
3. Click **Start Processing**
4. Use the chat, voice, quiz, summary, and email features

---

## 📁 File Overview

| File                        | Description                                      |
|-----------------------------|--------------------------------------------------|
| `streamlit_app_final.py`   | Main Streamlit interface                        |
| `picone.py`                | Downloads audio, transcribes, uploads to Pinecone |
| `chat_with_video.py`       | QA logic using LangChain                        |
| `chat_with_video_voice.py` | Voice-enabled QA (speech to text)               |
| `quiz_generator.py`        | Generates MCQs from transcript                  |
| `keyword_explorer.py`      | Extracts & links top keywords                   |
| `summary_and_email.py`     | Summarizes video & sends PDF via email          |
| `Conversational_RAG_Agent.py` | Optional RAG-based agent interface            |

---

## 👤 Contact

Developed by [Amjad Alo](https://github.com/AmjadAlo)  
📫 For issues, suggestions, or collaboration, feel free to reach out via GitHub.

## 🎥 Demo

- 📌 [Demo 1 –5min_video](https://drive.google.com/file/d/1xZZkoffVqn7h5GwBOcc-eQ7Q0igb-qaR/view?usp=drive_link)
- 📌 [Demo 2 – 5min_video](https://drive.google.com/file/d/1u_V05TL0HGi1f1H7-D4ypmqgVB_iPZUP/view?usp=drive_link)
- 📌 [Demo 3 – 5min_video](https://drive.google.com/file/d/1zd2WUwRYWawYvYEvd6eSS2KU8nzBa1EO/view?usp=drive_link)


