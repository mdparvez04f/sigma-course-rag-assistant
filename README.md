# Speech-to-Text RAG System for Video Content

A RAG-based question answering system that converts 
YouTube video audio into a searchable knowledge base 
using OpenAI GPT-4.

##  Why I Built This

I wanted to make video content searchable. Instead of 
watching long YouTube videos to find specific information,
I built a system where you can just ask questions and get 
answers directly from the video content.

##  What This Project Does

I took 10 YouTube videos from Code with Harry's Sigma 
Web Course, converted the audio to text, and built a 
RAG pipeline on top of it. Now instead of watching the 
videos, you can ask any question and the system finds 
the relevant part and gives you an accurate answer.

##  How It Works

1. Downloaded YouTube videos and converted to MP3
2. MP3 audio files converted to text using 
   Speech Recognition
3. Transcribed text split into smaller chunks
4. Each chunk converted to vector embeddings 
   using OpenAI Embeddings
5. Embeddings stored in a vector database
6. User asks a question
7. Question converted to embedding and matched 
   against stored chunks
8. Most relevant chunks retrieved
9. GPT-4 generates accurate answer from 
   retrieved chunks

##  Technologies Used

- Python
- OpenAI API (GPT-4)
- OpenAI Embeddings
- Speech Recognition
- Vector Database
- Semantic Search
- Joblib
- RAG Architecture

##  Project Structure

rag-video-assistant/
│
├── app.py                 # Main application
├── audio_converter.py     # MP3 to text conversion
├── chunker.py             # Text chunking
├── embeddings.py          # Embedding generation
├── retriever.py           # Vector search
├── requirements.txt       # Dependencies
└── README.md

##  How To Run

1. Clone the repository
https://github.com/mdparvez04f/sigma-course-rag-assistant

2. Install dependencies
pip install -r requirements.txt

3. Add your OpenAI API key in .env file
OPENAI_API_KEY=your_key_here

4. Run the application
python app.py

##  Key Learnings

- Learned how to process and transcribe audio 
  files using speech recognition
- Understood chunking strategies and how they 
  affect retrieval accuracy
- Gained hands-on experience with OpenAI 
  Embeddings and GPT-4 API
- Built a complete end-to-end RAG pipeline 
  from scratch

##  Future Improvements

- Directly take YouTube URL as input instead 
  of manual download
- Support multiple courses and channels
- Add conversation memory for follow-up questions
- Deploy as a web application
- Experiment with open-source models to 
  reduce API costs
