<div align="center">

![ClipWise Banner](https://github.com/user-attachments/assets/f680046c-5d7b-4e27-b84d-23afc5d6edaf)

# ClipWise - YouTube RAG Study Assistant

**An AI-powered web application that transforms any YouTube video into an interactive study companion.**

</div>

## Project Owners

| Name | Social |
|------|--------|
| **Ata Güneş** | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AtaGn) |
| **Esra Cesur** | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/EsraCesur4) |


## Video Tutorial

**Watch the complete tutorial on how to use ClipWise and explore all features:**

<div align="center">

[![ClipWise Tutorial](https://img.youtube.com/vi/fhtBpl73huU/maxresdefault.jpg)](https://www.youtube.com/watch?v=fhtBpl73huU)

**[Watch Tutorial: How to Use ClipWise](https://www.youtube.com/watch?v=fhtBpl73huU)**

</div>

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Frontend Design](#frontend-design)
- [API Configuration](#api-configuration)

---

## Overview

ClipWise is a Retrieval-Augmented Generation (RAG) based system that allows users to interact with YouTube videos by asking questions, generating quizzes, creating flashcards, and producing study notes. The application leverages cutting-edge AI technologies to make video-based learning more efficient and interactive.

**Core Capabilities:**    
🔹Interactive Q&A with video content    
🔹Automated quiz generation    
🔹Flashcard creation for key concepts    
🔹Comprehensive study notes generation    

## Key Features

🔸 **Transcription:** Converts video audio into accurate text using OpenAI Whisper, making content searchable and analyzable.
🔸 **Semantic Search:** Retrieves the most relevant transcript segments using ChromaDB vector database for precise context matching.
🔸 **AI Chat Assistant:** Provides contextual answers to questions using the Groq API powered by LLaMA 3.3 (70B parameters).
🔸 **Quiz Generator:** Creates multiple-choice questions automatically from video content to test understanding.
🔸 **Flashcards:** Generates interactive flashcards highlighting key points and concepts from the video.
🔸 **Study Notes:** Produces structured, summarized learning notes organized by topics covered in the video.

---


## Dataset Information

**Source:** User-provided YouTube videos

**Data Types:**
- Audio files (`.mp3` / `.wav`)
- Transcript text
- Semantic embeddings

**Storage Structure:**
- `youtube_rag.db` — SQLite database for video info, transcripts, and chat history
- `chroma_db/` — ChromaDB persistent directory for vector embeddings

**Note:** The dataset is generated dynamically during runtime. No pre-existing dataset is included.

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

<div align="center">

![Architecture Workflow](https://github.com/user-attachments/assets/68f352ec-b4ad-4fc9-af3e-bdada468cdb9)

</div>

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

## Installation

For complete installation instructions, system requirements, and troubleshooting guide, please refer to:

**[ClipWise Setup Guide.pdf](ClipWise%20Setup%20Guide.pdf)**

---

## Frontend Design

<div align="center">

![Frontend Screenshot](https://github.com/user-attachments/assets/88a03eee-66f1-4910-a6a1-bf6300bb5875)

</div>

---

## API Configuration

ClipWise requires a Groq API key to enable AI-powered features including chat responses, quiz generation, and flashcard creation.

**How to obtain your API key:**

1. Visit [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up or log in to your account
3. Generate a new API key
4. Copy the key and paste it into the sidebar field in ClipWise

**Note:** The API key is stored locally in your browser and is required for all AI-powered features.


