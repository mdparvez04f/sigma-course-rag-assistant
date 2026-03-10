import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from openai import OpenAI

# ====== CONFIG ======
OPENAI_API_KEY = ("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")
client = OpenAI(api_key=OPENAI_API_KEY)
json_folder = "jsons"
joblib_file = "embeddings.joblib"
top_results = 5
# ===================

# ====== Step 1: Load or create embeddings ======
if os.path.exists(joblib_file):
    print(f"Loading existing embeddings from {joblib_file}...")
    df = joblib.load(joblib_file)
else:
    print("Creating embeddings from JSON files...")
    records = []
    chunk_id = 0

    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file)) as f:
                data = json.load(f)
            for chunk in data["chunks"]:
                chunk["chunk_id"] = chunk_id
                records.append(chunk)
                chunk_id += 1

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} chunks.")

    # Create embeddings using OpenAI
    def create_embeddings(text_list):
        embeddings = []
        batch_size = 50  # process in batches to avoid large requests
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    print("Generating embeddings (this may take a few minutes)...")
    df["embedding"] = create_embeddings(df["text"].tolist())

    # Save for future use
    joblib.dump(df, joblib_file)
    print(f"Saved embeddings to {joblib_file} ✅")

# ====== Step 2: Ask user question ======
incoming_query = input("\nAsk a Question: ")

# Create embedding for the query
question_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=[incoming_query]
).data[0].embedding

# ====== Step 3: Compute similarities ======
embedding_matrix = np.vstack(df["embedding"])
similarities = cosine_similarity(embedding_matrix, [question_embedding]).flatten()

max_indx = similarities.argsort()[::-1][:top_results]
top_chunks = df.loc[max_indx]

# ====== Step 4: Build prompt for LLM ======
prompt = f"""
I am teaching web development in my Sigma web development course.

Here are video subtitle chunks containing video title, video number,
start time in seconds, end time in seconds, and the text spoken:

{top_chunks[['title', 'number', 'start', 'end', 'text']].to_json(orient='records')}

---------------------------------

User Question:
"{incoming_query}"

Answer the question naturally like a human teacher.
Tell the user which video and timestamp explains the topic.
If the question is unrelated to the course, say you can only answer course-related questions.
"""

# Save prompt for reference
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# ====== Step 5: Generate LLM response ======
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

answer = response.choices[0].message.content

print("\nAI Response:\n")
print(answer)

# Save response
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(answer)