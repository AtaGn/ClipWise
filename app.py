from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yt_dlp
import whisper
import os
import re
import json
from pathlib import Path
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime
import sqlite3
from langdetect import detect

app = Flask(__name__)
CORS(app)

# Initialize models
print("Loading models...")
# Use 'tiny' model for 5x faster transcription (change to 'base' for better accuracy)
whisper_model = whisper.load_model("tiny", device="cpu")  
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good quality
print("Models loaded! Using 'tiny' Whisper model for speed.")

# Initialize ChromaDB for vector storage (new API)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Directory for temporary audio files
AUDIO_DIR = Path("temp_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Initialize SQLite database for chat history
DB_PATH = "youtube_rag.db"

def init_database():
    """Initialize SQLite database for storing transcripts and chat history"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Videos table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            duration INTEGER,
            language TEXT,
            transcript TEXT,
            created_at TIMESTAMP
        )
    ''')
    
    # Chunks table with timestamps
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            chunk_text TEXT,
            start_time REAL,
            end_time REAL,
            chunk_index INTEGER,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
    ''')
    
    # Chat history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            question TEXT,
            answer TEXT,
            relevant_chunks TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
    ''')
    
    conn.commit()
    conn.close()

init_database()


def download_audio(youtube_url, output_path):
    """Download audio from YouTube video"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'http_chunk_size': 10485760,
        'retries': 10,
        'fragment_retries': 10,
        'skip_unavailable_fragments': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'video_id': info.get('id', '')
        }


def transcribe_audio_with_timestamps(audio_path):
    """Transcribe audio using Whisper with word-level timestamps - optimized"""
    result = whisper_model.transcribe(
        str(audio_path),
        word_timestamps=True,
        verbose=False,
        fp16=False,  # Use FP32 for CPU (more compatible)
        language=None,  # Auto-detect
        condition_on_previous_text=False,  # Faster
    )
    return result


def chunk_text_with_timestamps(segments, chunk_duration=30):
    """Split transcript into chunks based on time duration"""
    chunks = []
    current_chunk = {
        'text': '',
        'start_time': 0,
        'end_time': 0,
        'words': []
    }
    
    for segment in segments:
        segment_start = segment['start']
        segment_end = segment['end']
        segment_text = segment['text']
        
        # If adding this segment exceeds chunk duration, save current chunk
        if current_chunk['text'] and (segment_end - current_chunk['start_time']) > chunk_duration:
            chunks.append(current_chunk)
            current_chunk = {
                'text': segment_text,
                'start_time': segment_start,
                'end_time': segment_end,
                'words': []
            }
        else:
            # Add to current chunk
            if not current_chunk['text']:
                current_chunk['start_time'] = segment_start
            current_chunk['text'] += ' ' + segment_text
            current_chunk['end_time'] = segment_end
    
    # Add the last chunk
    if current_chunk['text']:
        chunks.append(current_chunk)
    
    return chunks


def create_vector_collection(video_id, chunks):
    """Create or update vector collection for a video"""
    try:
        collection = chroma_client.get_or_create_collection(
            name=f"video_{video_id}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Generate embeddings for all chunks
        texts = [chunk['text'].strip() for chunk in chunks]
        embeddings = embedding_model.encode(texts).tolist()
        
        # Store in ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            'start_time': chunk['start_time'],
            'end_time': chunk['end_time'],
            'chunk_index': i
        } for i, chunk in enumerate(chunks)]
        
        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        return True
    except Exception as e:
        print(f"Error creating vector collection: {e}")
        return False


def semantic_search(video_id, query, top_k=4):
    """Perform semantic search using embeddings"""
    try:
        collection = chroma_client.get_collection(name=f"video_{video_id}")
        query_embedding = embedding_model.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count())  # Don't request more than available
        )
        
        if not results['documents'] or not results['documents'][0]:
            print(f"No results returned from ChromaDB for video {video_id}")
            return []
        
        chunks = []
        for i in range(len(results['documents'][0])):
            chunks.append({
                'text': results['documents'][0][i],
                'start_time': results['metadatas'][0][i]['start_time'],
                'end_time': results['metadatas'][0][i]['end_time'],
                'chunk_index': results['metadatas'][0][i]['chunk_index'],
                'similarity': 1 - results['distances'][0][i] if results['distances'][0][i] else 1.0
            })
        
        print(f"Found {len(chunks)} chunks for query: {query}")
        return chunks
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        # Fallback to keyword search if semantic search fails
        try:
            return fallback_keyword_search(video_id, query, top_k)
        except:
            return []


def fallback_keyword_search(video_id, query, top_k=4):
    """Fallback keyword-based search when semantic search fails"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT chunk_text, start_time, end_time, chunk_index FROM chunks WHERE video_id = ?', (video_id,))
    all_chunks = cursor.fetchall()
    conn.close()
    
    if not all_chunks:
        return []
    
    # Simple keyword matching
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk_text, start_time, end_time, chunk_index in all_chunks:
        chunk_words = set(chunk_text.lower().split())
        matches = len(query_words & chunk_words)
        
        if matches > 0:
            scored_chunks.append({
                'text': chunk_text,
                'start_time': start_time,
                'end_time': end_time,
                'chunk_index': chunk_index,
                'similarity': matches / len(query_words)
            })
    
    # Sort by score and return top_k
    scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    return scored_chunks[:top_k]


def format_timestamp(seconds):
    """Convert seconds to MM:SS or HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def save_to_database(video_id, title, duration, language, transcript, chunks):
    """Save video and chunks to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Save video
    cursor.execute('''
        INSERT OR REPLACE INTO videos (video_id, title, duration, language, transcript, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (video_id, title, duration, language, transcript, datetime.now()))
    
    # Save chunks
    cursor.execute('DELETE FROM chunks WHERE video_id = ?', (video_id,))
    for i, chunk in enumerate(chunks):
        cursor.execute('''
            INSERT INTO chunks (video_id, chunk_text, start_time, end_time, chunk_index)
            VALUES (?, ?, ?, ?, ?)
        ''', (video_id, chunk['text'], chunk['start_time'], chunk['end_time'], i))
    
    conn.commit()
    conn.close()


def get_video_from_db(video_id):
    """Retrieve video from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM videos WHERE video_id = ?', (video_id,))
    video = cursor.fetchone()
    
    if video:
        cursor.execute('SELECT * FROM chunks WHERE video_id = ? ORDER BY chunk_index', (video_id,))
        chunks_data = cursor.fetchall()
        
        chunks = [{
            'text': chunk[2],
            'start_time': chunk[3],
            'end_time': chunk[4],
            'chunk_index': chunk[5]
        } for chunk in chunks_data]
        
        conn.close()
        return {
            'video_id': video[0],
            'title': video[1],
            'duration': video[2],
            'language': video[3],
            'transcript': video[4],
            'chunks': chunks
        }
    
    conn.close()
    return None


def save_chat_history(video_id, question, answer, relevant_chunks):
    """Save chat interaction to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_history (video_id, question, answer, relevant_chunks, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (video_id, question, answer, json.dumps(relevant_chunks), datetime.now()))
    
    conn.commit()
    conn.close()


def get_chat_history(video_id):
    """Retrieve chat history for a video"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT question, answer, relevant_chunks, timestamp 
        FROM chat_history 
        WHERE video_id = ? 
        ORDER BY timestamp DESC
    ''', (video_id,))
    
    history = cursor.fetchall()
    conn.close()
    
    return [{
        'question': h[0],
        'answer': h[1],
        'relevant_chunks': json.loads(h[2]),
        'timestamp': h[3]
    } for h in history]


@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('.', 'frontend.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets like logo"""
    return send_from_directory('assets', filename)

@app.route('/api/transcribe', methods=['POST'])
def transcribe_video():
    """Endpoint to transcribe a YouTube video"""
    data = request.json
    youtube_url = data.get('url')
    
    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        # Extract video ID
        video_id_match = re.search(r'(?:v=|youtu\.be/)([^&\s?]+)', youtube_url)
        if not video_id_match:
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        video_id = video_id_match.group(1)
        video_id = re.sub(r'[?&].*', '', video_id)
        
        # Check if already in database
        existing_video = get_video_from_db(video_id)
        if existing_video:
            return jsonify({
                'video_id': video_id,
                'title': existing_video['title'],
                'transcript': existing_video['transcript'],
                'chunks': existing_video['chunks'],
                'language': existing_video['language'],
                'cached': True
            })
        
        # Download audio
        audio_path = AUDIO_DIR / video_id
        print(f"Downloading audio from {youtube_url}...")
        video_info = download_audio(youtube_url, audio_path)
        
        # Get the actual audio file path
        audio_file = AUDIO_DIR / f"{video_id}.wav"
        
        # Check if file exists
        if not audio_file.exists():
            # Try finding any audio file with this video_id
            possible_files = list(AUDIO_DIR.glob(f"{video_id}.*"))
            if possible_files:
                audio_file = possible_files[0]
            else:
                raise Exception(f"Audio file not found after download")
            
        print(f"Transcribing audio: {audio_file.name}")
        result = transcribe_audio_with_timestamps(audio_file)
        
        # Detect language
        transcript_text = result['text']
        try:
            language = detect(transcript_text)
        except:
            language = result.get('language', 'unknown')
        
        # Create chunks with timestamps
        chunks = chunk_text_with_timestamps(result['segments'], chunk_duration=30)
        
        # Clean up audio file
        if audio_file.exists():
            audio_file.unlink()
        
        # Save to database
        save_to_database(video_id, video_info['title'], video_info['duration'], 
                        language, transcript_text, chunks)
        
        # Create vector embeddings
        print(f"Creating vector embeddings for {len(chunks)} chunks...")
        success = create_vector_collection(video_id, chunks)
        
        if not success:
            print(f"Warning: Failed to create vector collection for {video_id}")
        else:
            print(f"Successfully created vector collection for {video_id}")
        
        return jsonify({
            'video_id': video_id,
            'title': video_info['title'],
            'transcript': transcript_text,
            'chunks': chunks,
            'language': language,
            'cached': False
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query_video():
    """Endpoint to query a transcribed video"""
    data = request.json
    video_id = data.get('video_id')
    query = data.get('query')
    groq_api_key = data.get('api_key', '')
    
    if not video_id or not query:
        return jsonify({'error': 'Missing video_id or query'}), 400
    
    # Get video from database
    video_data = get_video_from_db(video_id)
    if not video_data:
        return jsonify({'error': 'Video not found. Please transcribe it first.'}), 404
    
    try:
        # Perform semantic search
        relevant_chunks = semantic_search(video_id, query, top_k=4)
        
        if not relevant_chunks:
            return jsonify({
                'answer': "I couldn't find relevant information in the transcript to answer your question."
            })
        
        # Prepare context with timestamps
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            timestamp = format_timestamp(chunk['start_time'])
            context_parts.append(f"[{timestamp}] {chunk['text']}")
        
        context = '\n\n'.join(context_parts)
        
        # Call Groq API
        language_name = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-cn': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish'
        }.get(video_data['language'], video_data['language'])
        
        payload = {
            'model': 'llama-3.3-70b-versatile',
            'messages': [{
                'role': 'user',
                'content': f'''You are analyzing a YouTube video titled "{video_data['title']}" which is in {language_name}.

Based on the following transcript excerpts (with timestamps), please answer the user's question.

Important: 
- The video content is in {language_name}, so avoid using terms like "the speaker says" or "the narrator mentions". Instead, use neutral terms like "the video shows", "according to the content", "the video discusses", etc.
- If the excerpts don't contain enough information, say so clearly and suggest the user try a different question.
- Be helpful and informative while being concise.

Transcript excerpts (ordered by relevance):
{context}

User question: {query}

Please provide a helpful answer. When relevant, mention the timestamps where information was found (e.g., "At 2:35...").'''
            }],
            'temperature': 0.5,  # Lower temperature for more focused answers
            'max_tokens': 1500,
        }
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {groq_api_key}'
            },
            json=payload
        )
        
        if response.status_code != 200:
            error_detail = response.json().get('error', {}).get('message', 'Unknown error')
            return jsonify({'error': f'Groq API error: {error_detail}'}), 500
        
        result = response.json()
        answer = result['choices'][0]['message']['content']
        
        # Save to chat history
        chunks_info = [{
            'text': c['text'][:100] + '...',
            'timestamp': format_timestamp(c['start_time']),
            'start_time': c['start_time']
        } for c in relevant_chunks]
        
        save_chat_history(video_id, query, answer, chunks_info)
        
        return jsonify({
            'answer': answer,
            'relevant_chunks': chunks_info,
            'chunk_count': len(relevant_chunks)
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<video_id>', methods=['GET'])
def get_history(video_id):
    """Get chat history for a video"""
    try:
        history = get_chat_history(video_id)
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-transcript', methods=['POST'])
def search_transcript():
    """Search within transcript"""
    data = request.json
    video_id = data.get('video_id')
    search_term = data.get('search_term', '').lower()
    
    if not video_id or not search_term:
        return jsonify({'error': 'Missing video_id or search_term'}), 400
    
    video_data = get_video_from_db(video_id)
    if not video_data:
        return jsonify({'error': 'Video not found'}), 404
    
    # Search through chunks
    results = []
    for chunk in video_data['chunks']:
        if search_term in chunk['text'].lower():
            results.append({
                'text': chunk['text'],
                'timestamp': format_timestamp(chunk['start_time']),
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time']
            })
    
    return jsonify({'results': results, 'count': len(results)})


@app.route('/api/study-buddy/quiz', methods=['POST'])
def generate_quiz():
    """Generate quiz questions from video content"""
    data = request.json
    video_id = data.get('video_id')
    num_questions = data.get('num_questions', 5)
    difficulty = data.get('difficulty', 'medium')
    groq_api_key = data.get('api_key', '')
    
    if not video_id:
        return jsonify({'error': 'Missing video_id'}), 400
    
    video_data = get_video_from_db(video_id)
    if not video_data:
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        # Get random chunks from the video
        import random
        sample_size = min(10, len(video_data['chunks']))
        sample_chunks = random.sample(video_data['chunks'], sample_size)
        context = '\n\n'.join([chunk['text'] for chunk in sample_chunks])
        
        payload = {
            'model': 'llama-3.3-70b-versatile',
            'messages': [{
                'role': 'user',
                'content': f'''Based on the following video content, generate {num_questions} multiple-choice quiz questions at {difficulty} difficulty level.

Video Title: {video_data['title']}

Content:
{context}

Generate questions in this EXACT JSON format (respond with ONLY valid JSON, no other text):
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": 0,
      "explanation": "Brief explanation of why this is correct"
    }}
  ]
}}

Important:
- Make questions test understanding, not just memorization
- Ensure questions are clear and unambiguous
- Make all options plausible
- correct_answer is the index (0-3) of the correct option
- RESPOND WITH ONLY VALID JSON, NO MARKDOWN OR OTHER TEXT'''
            }],
            'temperature': 0.7,
            'max_tokens': 2000,
        }
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {groq_api_key}'
            },
            json=payload
        )
        
        if response.status_code != 200:
            error_detail = response.json().get('error', {}).get('message', 'Unknown error')
            return jsonify({'error': f'API error: {error_detail}'}), 500
        
        result = response.json()
        answer_text = result['choices'][0]['message']['content']
        
        # Clean up response - remove markdown if present
        answer_text = answer_text.strip()
        if answer_text.startswith('```json'):
            answer_text = answer_text.split('```json')[1].split('```')[0].strip()
        elif answer_text.startswith('```'):
            answer_text = answer_text.split('```')[1].split('```')[0].strip()
        
        quiz_data = json.loads(answer_text)
        
        return jsonify(quiz_data)
        
    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/study-buddy/flashcards', methods=['POST'])
def generate_flashcards():
    """Generate flashcards from video content"""
    data = request.json
    video_id = data.get('video_id')
    num_cards = data.get('num_cards', 10)
    groq_api_key = data.get('api_key', '')
    
    if not video_id:
        return jsonify({'error': 'Missing video_id'}), 400
    
    video_data = get_video_from_db(video_id)
    if not video_data:
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        # Get sample of content
        import random
        sample_size = min(15, len(video_data['chunks']))
        sample_chunks = random.sample(video_data['chunks'], sample_size)
        context = '\n\n'.join([chunk['text'] for chunk in sample_chunks])
        
        payload = {
            'model': 'llama-3.3-70b-versatile',
            'messages': [{
                'role': 'user',
                'content': f'''Based on the following video content, create {num_cards} flashcards for studying.

Video Title: {video_data['title']}

Content:
{context}

Generate flashcards in this EXACT JSON format (respond with ONLY valid JSON):
{{
  "flashcards": [
    {{
      "front": "Question or concept",
      "back": "Answer or explanation",
      "category": "Category/topic"
    }}
  ]
}}

Important:
- Focus on key concepts, definitions, and important facts
- Keep front concise (question or prompt)
- Make back clear and informative
- Include relevant categories
- RESPOND WITH ONLY VALID JSON, NO MARKDOWN'''
            }],
            'temperature': 0.7,
            'max_tokens': 2000,
        }
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {groq_api_key}'
            },
            json=payload
        )
        
        if response.status_code != 200:
            error_detail = response.json().get('error', {}).get('message', 'Unknown error')
            return jsonify({'error': f'API error: {error_detail}'}), 500
        
        result = response.json()
        answer_text = result['choices'][0]['message']['content'].strip()
        
        # Clean markdown
        if answer_text.startswith('```json'):
            answer_text = answer_text.split('```json')[1].split('```')[0].strip()
        elif answer_text.startswith('```'):
            answer_text = answer_text.split('```')[1].split('```')[0].strip()
        
        cards_data = json.loads(answer_text)
        
        return jsonify(cards_data)
        
    except Exception as e:
        print(f"Error generating flashcards: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/study-buddy/notes', methods=['POST'])
def generate_study_notes():
    """Generate comprehensive study notes from video"""
    data = request.json
    video_id = data.get('video_id')
    groq_api_key = data.get('api_key', '')
    
    if not video_id:
        return jsonify({'error': 'Missing video_id'}), 400
    
    video_data = get_video_from_db(video_id)
    if not video_data:
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        # Use full transcript for notes
        transcript = video_data['transcript']
        
        payload = {
            'model': 'llama-3.3-70b-versatile',
            'messages': [{
                'role': 'user',
                'content': f'''Create comprehensive study notes from this video transcript.

Video Title: {video_data['title']}

Transcript:
{transcript[:8000]}  # Limit to avoid token limits

Create structured study notes with:
1. Main Topics/Themes
2. Key Concepts & Definitions
3. Important Facts & Details
4. Summary/Conclusion

Format as clear, organized notes suitable for studying.'''
            }],
            'temperature': 0.5,
            'max_tokens': 2000,
        }
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {groq_api_key}'
            },
            json=payload
        )
        
        if response.status_code != 200:
            error_detail = response.json().get('error', {}).get('message', 'Unknown error')
            return jsonify({'error': f'API error: {error_detail}'}), 500
        
        result = response.json()
        notes = result['choices'][0]['message']['content']
        
        return jsonify({'notes': notes})
        
    except Exception as e:
        print(f"Error generating notes: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check server status"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM videos')
    video_count = cursor.fetchone()[0]
    conn.close()
    
    return jsonify({
        'status': 'online',
        'model': 'whisper-tiny',
        'embedding_model': 'all-MiniLM-L6-v2',
        'cached_videos': video_count
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("YouTube RAG Backend Server - Enhanced")
    print("="*50)
    print(f"Server running on: http://localhost:5000")
    print(f"Whisper model: base")
    print(f"Embedding model: all-MiniLM-L6-v2")
    print(f"Database: {DB_PATH}")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)