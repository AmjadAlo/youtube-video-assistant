# keywords_tool.py

import re
import streamlit as st
from summary_and_email import load_transcript
from keybert import KeyBERT




# ------------------------------------------------------------------------
# feat: extract top keywords using KeyBERT
# ------------------------------------------------------------------------
def extract_keywords(text, num_keywords=5):
    kw_model = KeyBERT()  # Initialize keyword extraction model
    keywords = kw_model.extract_keywords(text, top_n=num_keywords, stop_words='english')  # Extract keywords
    return [kw[0] for kw in keywords]  # Return only keyword strings





# ------------------------------------------------------------------------
# ui: Streamlit interface to explore keywords from video transcript
# ------------------------------------------------------------------------
def keyword_explorer():
    st.markdown("### Top 5 Keywords from Video")

    try:
        # Load transcript from source
        transcript, _ = load_transcript()

        # Clean up whitespace for better keyword extraction
        clean_text = re.sub(r'\s+', ' ', transcript)

        # Extract top 5 keywords
        keywords = extract_keywords(clean_text)

        # Display keywords as clickable Wikipedia links
        for word in keywords:
            link = f"https://en.wikipedia.org/wiki/{word.replace(' ', '_')}"
            st.markdown(f"- [{word.title()}]({link})")

    except Exception as e:
        # Handle and display errors in Streamlit UI
        st.error(f"Failed to extract keywords: {e}")
