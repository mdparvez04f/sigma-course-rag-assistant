from openai import OpenAI
import os
import json

client = OpenAI(api_key="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")


json_folder = "jsons"

for file in os.listdir(json_folder):

    if file.endswith(".json"):

        path = os.path.join(json_folder, file)

        with open(path, "r") as f:
            data = json.load(f)

        chunks = data["chunks"]

        texts = [chunk["text"] for chunk in chunks]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        for i, emb in enumerate(response.data):
            chunks[i]["embedding"] = emb.embedding

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print("Embeddings added to:", file)