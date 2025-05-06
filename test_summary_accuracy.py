from datasets import load_dataset
from evaluate import load
from summary_and_email import summarize_transcript

# Load test data (10 samples from CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")
rouge = load("rouge")

# Get the transcript and reference summary
inputs = dataset["article"]
references = dataset["highlights"]

# Generate summaries with your model
predictions = []
for i, text in enumerate(inputs):
    print(f"Summarizing sample {i+1}/10...")
    try:
        summary = summarize_transcript(text)
        predictions.append(summary)
    except Exception as e:
        print(f"Error: {e}")
        predictions.append("")

# Evaluate
results = rouge.compute(predictions=predictions, references=references)
print("\n ROUGE Evaluation:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
