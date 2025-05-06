import os
import re
import smtplib
import requests
from email.message import EmailMessage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from io import BytesIO

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI



# ------------------------------------------------------------------------
# config: Securely retrieve OpenAI API key from environment
# ------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 🔐 Avoid hardcoding API keys


# ------------------------------------------------------------------------
# feat: Load transcript and namespace from local file
# ------------------------------------------------------------------------
def load_transcript():
    with open("current_namespace.txt", "r", encoding="utf-8") as f:
        namespace = f.read().strip()

    # Find transcript file using normalized namespace
    transcript_path = None
    for f in os.listdir("data"):
        if f.lower() == f"{namespace}_transcription.txt".lower():
            transcript_path = os.path.join("data", f)
            break

    if not transcript_path:
        raise FileNotFoundError("Transcript file not found.")

    with open(transcript_path, "r", encoding="utf-8") as f:
        return f.read(), namespace




# ------------------------------------------------------------------------
# feat: Summarize a transcript using LLM (5–7 sentence summary)
# ------------------------------------------------------------------------
def summarize_transcript(transcript_text):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Summarize the transcript in 5-7 sentences using only the information from the transcript."
        ),
        HumanMessagePromptTemplate.from_template("Transcript:\n{transcript}\n\nSummary:")
    ])

    # Use OpenAI's GPT model with moderate creativity
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, openai_api_key=OPENAI_API_KEY)
    chain = prompt | llm

    # Limit input to 4000 chars to avoid token overflow
    return chain.invoke({"transcript": transcript_text[:4000]}).content



# ------------------------------------------------------------------------
# feat: Fetch related image from Unsplash (no key required)
# ------------------------------------------------------------------------
def fetch_related_image(query):
    try:
        # Public endpoint for random image by keyword
        response = requests.get(f"https://source.unsplash.com/800x400/?{query}", timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception:
        return None  # Silent fail if image can’t be fetched




# ------------------------------------------------------------------------
# util: Word-wrap long summary lines for PDF layout
# ------------------------------------------------------------------------
def wrap_text(text, max_chars=90):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line + word) <= max_chars:
            line += word + " "
        else:
            lines.append(line.strip())
            line = word + " "
    if line:
        lines.append(line.strip())
    return lines




# ------------------------------------------------------------------------
# feat: Generate a PDF with title, optional image, and text summary
# ------------------------------------------------------------------------
def generate_pdf(summary_text, title, image=None):
    # Normalize filename
    safe_title = re.sub(r"[^\w\s]", "", title).replace(" ", "_")
    pdf_path = f"{safe_title}.pdf"

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Add title
    y = height - 75
    c.setFont("Helvetica-Bold", 20)
    c.drawString(80, y, title)
    y -= 30

    # Add image if provided
    if image:
        try:
            image = image.resize((500, 250))
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            c.drawImage(ImageReader(img_buffer), 50, y - 250)
            y -= 270
        except Exception:
            pass  # Skip image if any issues occur

    # Add summary text (wrapped to page width)
    c.setFont("Helvetica", 12)
    wrapped_lines = wrap_text(summary_text, max_chars=100)
    for line in wrapped_lines:
        if y < 50:  # Add page break if needed
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)
        c.drawString(50, y, line)
        y -= 20

    c.save()
    return pdf_path




# ------------------------------------------------------------------------
# feat: Send PDF file via email using Gmail SMTP and app password
# ------------------------------------------------------------------------
def send_email_with_pdf(pdf_path, recipient_emails, sender_email, app_password):
    sender_email = sender_email.strip()
    app_password = app_password.strip()
    recipient_emails = [e.strip() for e in recipient_emails]

    # Create email content
    msg = EmailMessage()
    msg["Subject"] = "Your Video Summary PDF"
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipient_emails)
    msg.set_content("Attached is the summary PDF.")

    # Attach PDF file
    with open(pdf_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=os.path.basename(pdf_path)
        )

    # Connect to Gmail SMTP and send
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        raise RuntimeError("❌ Authentication failed. Please check your email and app password.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to send email: {type(e).__name__} - {e}")
