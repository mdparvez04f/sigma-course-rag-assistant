from openai import OpenAI
import os
import json

client = OpenAI(api_key="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")


audio_folder = "audios"
json_folder = "jsons"

os.makedirs(json_folder, exist_ok=True)

audio_files = os.listdir(audio_folder)

for audio in audio_files:
    if audio.endswith(".mp3"):

        number = audio.split("_")[1].split(".")[0]
        title = f"audio_{number}"

        print("Processing:", audio)

        with open(f"{audio_folder}/{audio}", "rb") as audio_file:

            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )

        hindi_text = transcription.text

        # Translate to English
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Translate this Hindi text to English:\n{hindi_text}"
        )

        english_text = response.output_text

        # Create chunks manually (sentence based)
        sentences = english_text.split(".")

        chunks = []
        start_time = 0

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence == "":
                continue

            chunk = {
                "number": number,
                "title": title,
                "start": start_time,
                "end": start_time + 5,   # dummy duration
                "text": sentence
            }

            chunks.append(chunk)
            start_time += 5

        chunks_with_metadata = {
            "chunks": chunks,
            "text": english_text
        }

        with open(f"{json_folder}/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f, indent=2)

print("All files processed")