from openai import OpenAI

client = OpenAI(api_key="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")

audio_file = open("audios/audio_1.mp3","rb")

transcription = client.audio.transcriptions.create(
    model="gpt-4o-transcribe",
    file=audio_file
)

hindi_text = transcription.text

response = client.responses.create(
    model="gpt-4.1-mini",
    input=f"Translate this Hindi text to English: {hindi_text}"
)

print(response.output_text)

