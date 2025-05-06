import os
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate



# ------------------------------------------------------------------------
# util: Normalize namespace and filenames from video title or ID
# ------------------------------------------------------------------------
def normalize_namespace(title):
    return re.sub(r"[^a-z0-9]+", "_", title.lower().replace("&", "and")).strip("_")




# ------------------------------------------------------------------------
# feat: Load transcript file based on current namespace
# ------------------------------------------------------------------------
def load_transcript():
    try:
        # Read the namespace used in the last pipeline run
        with open("current_namespace.txt", "r", encoding="utf-8") as f:
            raw_namespace = f.read().strip()

        namespace = normalize_namespace(raw_namespace)
        expected_filename = f"{namespace}_transcription.txt"
        data_dir = "data"

        # Search for the exact file in the /data directory
        for filename in os.listdir(data_dir):
            if filename.lower() == expected_filename.lower():
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    return f.read()

        raise FileNotFoundError(f"No matching transcript file found for: {expected_filename}")

    except Exception as e:
        raise FileNotFoundError(f"Could not load transcript: {e}")




# ------------------------------------------------------------------------
# feat: Generate quiz questions from transcript using OpenAI LLM
# ------------------------------------------------------------------------
def generate_quiz_questions(transcript_text, num_questions=5):
    # Define prompt template for the quiz generator
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a quiz generator. Only use the transcript provided. "
            "Create {n} multiple-choice questions with four options (A, B, C, D). Include the correct answer for each."
        ),
        HumanMessagePromptTemplate.from_template(
            "Transcript:\n{transcript}\n\nGenerate {n} multiple-choice questions."
        )
    ])

    # Initialize LLM with moderate creativity
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")  # 🔐 Use environment variable for safety
    )

    # Format the input prompt (truncate if needed)
    formatted_prompt = prompt.format_messages(transcript=transcript_text[:4000], n=num_questions)

    # Debug: show prompt sent to the LLM
    print("\nPrompt sent to LLM >>>")
    for m in formatted_prompt:
        print(m)

    try:
        # Send prompt to OpenAI
        response = llm.invoke(formatted_prompt)

        # Debug: display the raw response
        print("\nLLM Response >>>\n", response.content)

        return response.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ""




# ------------------------------------------------------------------------
# feat: Parse quiz response text into structured question format
# ------------------------------------------------------------------------
def parse_questions(text):
    questions = []

    # Split based on lines starting with a number and period (e.g., "1. ")
    blocks = re.split(r"\n\s*\d+\.\s", text)

    for block in blocks[1:]:  # skip intro section if present
        lines = block.strip().split("\n")
        if len(lines) < 6:
            continue

        # Extract question line
        q_line = lines[0].strip()

        # Extract up to 4 choices (e.g., "A) text", "B. text")
        choices = []
        for line in lines[1:]:
            if re.match(r"^[A-D]\)", line.strip()) or re.match(r"^[A-D]\.", line.strip()):
                choices.append(line.strip())
            if len(choices) == 4:
                break

        # Try to find the correct answer (e.g., "Correct answer: A)")
        correct_line = next((line for line in lines if "correct" in line.lower()), "")
        correct_match = re.search(r"([A-D])\)", correct_line)
        correct_letter = correct_match.group(1) if correct_match else ""

        # Only append if all parts are valid
        if q_line and len(choices) == 4 and correct_letter:
            questions.append((q_line, choices, correct_letter))

    return questions
