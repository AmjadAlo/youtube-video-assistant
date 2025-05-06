import os
import re
import json
import subprocess
from pathlib import Path

from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import whisper


# ------------------------------------------------------------------------
# config: Load API keys and configuration from environment or defaults
# ------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")                  # Pinecone API key (required)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "youtube-video-index")  # Vector DB index name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                      # OpenAI key (optional for Whisper large)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"                         # HuggingFace embedding model
PINECONE_CLOUD = "aws"                                            # Pinecone cloud provider
PINECONE_REGION = "us-east-1"                                     # Pinecone region for serverless

# Optional LangSmith observability config
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<LANGCHAIN_API_KEY>"          # Replace externally
os.environ["LANGCHAIN_PROJECT"] = "pr-grumpy-simple-26"




# ------------------------------------------------------------------------
# feat: Download audio from YouTube video and extract video metadata
# ------------------------------------------------------------------------
def download_audio_from_video(video_url: str) -> tuple:
    try:
        # Run yt-dlp to get video info without downloading content
        result = subprocess.run(
            ["yt-dlp", "--no-playlist", "--skip-download", "--print-json", video_url],
            capture_output=True, text=True, check=True
        )
        video_info = json.loads(result.stdout)
        title = video_info.get("title", "downloaded_audio")  # fallback title if missing
    except Exception as e:
        raise RuntimeError(f"Failed to fetch video info: {e}")

    # refactor: Sanitize title for file-safe names
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).replace(" ", "_")
    output_filename = f"{safe_title}.mp3"

    # feat: Extract audio using yt-dlp and convert to MP3
    command = [
        "yt-dlp", "--no-playlist",
        "--extract-audio", "--audio-format", "mp3", "--audio-quality", "7",
        "-o", output_filename, video_url
    ]
    subprocess.run(command, check=True)  # raises error if command fails
    print(f"Audio saved as: {output_filename}")

    # feat: Save key metadata to JSON file
    metadata = {
        "title": video_info.get("title"),
        "description": video_info.get("description"),
        "uploader": video_info.get("uploader"),
        "upload_date": video_info.get("upload_date"),
        "duration": video_info.get("duration"),
        "view_count": video_info.get("view_count"),
        "like_count": video_info.get("like_count"),
        "categories": video_info.get("categories"),
        "tags": video_info.get("tags"),
        "url": video_url
    }

    os.makedirs("data", exist_ok=True)  # create folder if not exists
    metadata_path = f"data/{safe_title}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {metadata_path}")

    return output_filename, safe_title





# ------------------------------------------------------------------------
# feat: Transcribe MP3 audio file using OpenAI Whisper
# ------------------------------------------------------------------------
def transcribe_audio(audio_path: str, output_text_path: str = "transcription.txt", model_size: str = "tiny") -> str:
    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)  # can be tiny/base/small/medium/large
    print(f"Transcribing audio file: {audio_path}...")

    result = model.transcribe(audio_path)  # returns a dict with 'text'

    # Save transcript to file
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Transcription completed: {output_text_path}")
    return result["text"]





# ------------------------------------------------------------------------
# feat: Split large transcript into overlapping text chunks
# ------------------------------------------------------------------------
def split_text_into_chunks(text_path: str, chunk_size: int = 400, chunk_overlap: int = 100) -> list:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Use LangChain's RecursiveCharacterTextSplitter to handle large text
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)

    print(f"Text split into {len(chunks)} chunks.")
    return chunks





# ------------------------------------------------------------------------
# feat: Embed transcript chunks and store them in Pinecone vector DB
# ------------------------------------------------------------------------
def embed_chunks_and_upload_to_pinecone(chunks: list, namespace: str):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # fix: Create index if it doesn't exist yet
    if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # matches embedding output
            metric="cosine",
            spec={"serverless": {"cloud": PINECONE_CLOUD, "region": PINECONE_REGION}}
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Using existing index '{PINECONE_INDEX_NAME}'.")

    # Initialize index and embedding model
    index = pc.Index(PINECONE_INDEX_NAME)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Loop through each chunk, embed, and upload with unique ID
    for idx, chunk in enumerate(chunks):
        vector_id = f"chunk-{idx}"
        vector = embeddings.embed_query(chunk)
        metadata = {"text": chunk}
        index.upsert(vectors=[(vector_id, vector, metadata)], namespace=namespace)

    print(f"Uploaded {len(chunks)} chunks to Pinecone (namespace='{namespace}').")





# ------------------------------------------------------------------------
# util: Clean and normalize titles for namespace use
# ------------------------------------------------------------------------
def normalize_namespace(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower().replace("&", "and")).strip("_")





# ------------------------------------------------------------------------
# fix: Ensure transcript file has a normalized name for future tools
# ------------------------------------------------------------------------
def normalize_transcript_filename():
    try:
        with open("current_namespace.txt", "r", encoding="utf-8") as f:
            raw_namespace = f.read().strip()
        normalized = normalize_namespace(raw_namespace)

        for filename in os.listdir("data"):
            if filename.endswith("_transcription.txt") and normalized in filename.lower():
                expected_name = f"{normalized}_transcription.txt"
                current_path = os.path.join("data", filename)
                expected_path = os.path.join("data", expected_name)

                if filename != expected_name:
                    os.rename(current_path, expected_path)
                    print(f" Renamed: {filename} → {expected_name}")
                return

        print(f"⚠️ No matching transcript file found for normalization: {normalized}")
    except Exception as e:
        print(f" Error normalizing transcript filename: {e}")





# ------------------------------------------------------------------------
# main: End-to-end video processing pipeline (audio → vector store)
# ------------------------------------------------------------------------
def main_workflow(video_url: str) -> str:
    print(" Running YouTube video pipeline...")

    # Step 1: Download video and extract metadata
    audio_path, safe_title = download_audio_from_video(video_url)

    # Step 2: Transcribe audio to text
    os.makedirs("data", exist_ok=True)
    raw_transcription_path = f"data/{safe_title}_transcription.txt"
    transcribe_audio(audio_path, output_text_path=raw_transcription_path)

    # Step 2.5: Normalize filename for tool compatibility
    normalized_title = normalize_namespace(safe_title)
    final_transcription_path = f"data/{normalized_title}_transcription.txt"
    if raw_transcription_path != final_transcription_path:
        os.rename(raw_transcription_path, final_transcription_path)

    # Step 3: Break long transcript into manageable chunks
    chunks = split_text_into_chunks(final_transcription_path)

    # Step 4: Embed chunks and upload to vector DB
    embed_chunks_and_upload_to_pinecone(chunks, namespace=normalized_title)

    # Step 5: Store current namespace for downstream tools
    with open("current_namespace.txt", "w", encoding="utf-8") as f:
        f.write(normalized_title)

    #  Summary message
    return f""" All steps completed!
Transcript saved to: {final_transcription_path}
Pinecone namespace: {normalized_title}
"""
