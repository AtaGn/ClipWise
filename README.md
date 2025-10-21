# ClipWise — YouTube RAG Study Assistant

<img width="3644" height="375" alt="QR kodu (2)" src="https://github.com/user-attachments/assets/b21be8ef-4e25-4b96-b2b0-381021dea580" />

An AI-powered web application that transforms any YouTube video into an interactive study companion.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Usage](#usage)
- [Technology Stack](#technology-stack)
- [API Configuration](#api-configuration)
- [How It Works](#how-it-works)

---

## Overview

ClipWise is a Retrieval-Augmented Generation (RAG) based system that allows users to interact with YouTube videos by asking questions, generating quizzes, creating flashcards, and producing study notes. The application leverages cutting-edge AI technologies to make video-based learning more efficient and interactive.

**Core Capabilities:**
- Interactive Q&A with video content
- Automated quiz generation
- Flashcard creation for key concepts
- Comprehensive study notes generation

---

## Key Features

### Transcription
Converts video audio into accurate text using OpenAI Whisper, making content searchable and analyzable.

### Semantic Search
Retrieves the most relevant transcript segments using ChromaDB vector database for precise context matching.

### AI Chat Assistant
Provides contextual answers to questions using the Groq API powered by LLaMA 3.3 (70B parameters).

### Quiz Generator
Creates multiple-choice questions automatically from video content to test understanding.

### Flashcards
Generates interactive flashcards highlighting key points and concepts from the video.

### Study Notes
Produces structured, summarized learning notes organized by topics covered in the video.

---

## System Architecture

ClipWise employs a multi-layered architecture that seamlessly integrates various AI technologies:

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | HTML, TailwindCSS, Vanilla JS | User interface and API integration |
| **Backend** | Flask (Python) | REST API and model orchestration |
| **Speech-to-Text** | Whisper (tiny model) | Converts video audio to text |
| **Embedding Model** | SentenceTransformer (all-MiniLM-L6-v2) | Creates semantic embeddings |
| **Vector Database** | ChromaDB | Performs semantic similarity search |
| **LLM API** | Groq (LLaMA 3.3 - 70B) | Generates contextual responses |
| **Database** | SQLite | Stores video data, transcripts, and chat history |

### Architecture Workflow

1. User provides a YouTube video link
2. `yt-dlp` downloads the audio
3. Whisper transcribes the content
4. Transcript is split into time-based segments
5. SentenceTransformer encodes each segment into embeddings
6. Embeddings are stored in ChromaDB
7. User queries are converted into embeddings
8. Relevant transcript chunks are retrieved
9. Groq API produces the final context-aware response

---

## Dataset Information

**Source:** User-provided YouTube videos

**Data Types:**
- Audio files (`.mp3` / `.wav`)
- Transcript text
- Semantic embeddings

**Processing Pipeline:**
1. Audio extraction using `yt-dlp`
2. Automatic transcription using OpenAI Whisper
3. Chunking transcripts into 30-second segments
4. Generating embeddings with SentenceTransformer

**Storage Structure:**
- `youtube_rag.db` — SQLite database for video info, transcripts, and chat history
- `chroma_db/` — ChromaDB persistent directory for vector embeddings

**Note:** The dataset is generated dynamically during runtime. No pre-existing dataset is included.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Internet connection for downloading models and processing videos

### Setup Steps

**1. Clone the repository**
```bash
git clone <repository-url>
cd clipwise
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the backend server**
```bash
python app.py
```

**5. Access the web interface**

Open your browser and navigate to:
```
http://localhost:5000
```

### Requirements

All dependencies are listed in `requirements.txt`:

```
flask==3.0.0
flask-cors==4.0.0
yt-dlp>=2025.1.1
openai-whisper==20231117
torch==2.1.2
torchaudio==2.1.2
numpy==1.24.3
requests>=2.32.2
sentence-transformers==2.6.1
transformers==4.39.3
chromadb==0.5.5
langdetect==1.0.9
tqdm>=4.66.3
pandas>=2.2.2
```

---

## Usage

### Getting Started

1. **Obtain API Key**: Visit [Groq Console](https://console.groq.com/keys) to get your free API key
2. **Enter API Key**: Input your key in the sidebar field
3. **Add Video**: Click "Add New Video" and paste a YouTube link
4. **Wait for Processing**: The system will download, transcribe, and index the video
5. **Start Learning**: Ask questions, generate quizzes, create flashcards, or produce study notes

### Interface Layout

**Left Sidebar:** API key entry and video history

**Center Panel:** Chat interface with options for quizzes, flashcards, and notes

**Right Panel:** YouTube player and transcript viewer

---

## Technology Stack

### Frontend
- HTML5
- TailwindCSS for responsive design
- JavaScript for dynamic interactions

<img width="1861" height="1277" alt="frontend" src="https://github.com/user-attachments/assets/475924ca-68b7-4aed-8649-ebbdd3eb819b" />


### Backend
- Flask web framework
- Python 3.10+

### AI & Machine Learning
- **Whisper**: Automatic speech recognition
- **SentenceTransformer**: Semantic embedding generation
- **ChromaDB**: Vector database for similarity search
- **Groq API**: Large language model inference

### Data Storage
- SQLite for structured data
- ChromaDB for vector embeddings

---

## API Configuration

ClipWise requires a Groq API key to enable AI-powered features including chat responses, quiz generation, and flashcard creation.

**How to obtain your API key:**

1. Visit [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up or log in to your account
3. Generate a new API key
4. Copy the key and paste it into the sidebar field in ClipWise

**Note:** The API key is stored locally in your browser and is required for all AI-powered features.

---

## How It Works

### Problem Statement
Finding key information in long educational YouTube videos is time-consuming and inefficient. Traditional methods require watching entire videos or manually searching through transcripts.

### Solution
ClipWise automates this process through transcription, semantic retrieval, and AI-driven question answering, allowing users to study interactively and efficiently. The RAG architecture ensures responses are grounded in actual video content rather than hallucinated information.

### Technical Implementation
**Step 1: Audio Extraction**
The system uses `yt-dlp` to download audio from YouTube videos in optimal quality.

**Step 2: Transcription**
OpenAI Whisper converts the audio into accurate text transcripts with timestamps.

**Step 3: Semantic Indexing**
Transcripts are chunked into 30-second segments and converted into vector embeddings using SentenceTransformer.

**Step 4: Vector Storage**
Embeddings are stored in ChromaDB for fast semantic similarity search.

**Step 5: Query Processing**
User questions are embedded and matched against the vector database to retrieve relevant context.

**Step 6: Response Generation**
The Groq API uses retrieved context to generate accurate, context-aware responses.

