# app.py
import os
import re
import time
import numpy as np
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

api = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI(api_key=api)

# Tiktoken encoding for token counting
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Count the number of tokens in a string."""
    return len(encoding.encode(text))

def extract_video_id(youtube_url):
    """Extract the YouTube video ID from a URL."""
    # Match patterns like youtu.be/VIDEO_ID or youtube.com/watch?v=VIDEO_ID
    regex_patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\?\/]+)',
        r'youtube\.com\/embed\/([^&\?\/]+)',
        r'youtube\.com\/v\/([^&\?\/]+)',
    ]
    
    for pattern in regex_patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    return None

def get_transcript(video_id):
    """Get the transcript for a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript_list
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def segment_transcript(transcript, max_tokens=250):
    """Segment the transcript into chunks of approximately max_tokens tokens."""
    segments = []
    current_segment = {"text": "", "start": None, "end": None}
    current_tokens = 0
    
    for entry in transcript:
        text = entry["text"]
        start_time = entry["start"]
        duration = entry["duration"]
        tokens_in_text = count_tokens(text)
        
        # If this is the first entry in a new segment
        if current_segment["start"] is None:
            current_segment["start"] = start_time
        
        # If adding this text would exceed max_tokens, save the current segment and start a new one
        if current_tokens + tokens_in_text > max_tokens and current_tokens > 0:
            current_segment["end"] = start_time
            segments.append(current_segment)
            current_segment = {"text": text, "start": start_time, "end": None}
            current_tokens = tokens_in_text
        else:
            # Otherwise, add this text to the current segment
            current_segment["text"] += " " + text
            current_tokens += tokens_in_text
    
    # Add the last segment if it's not empty
    if current_segment["text"] and current_segment["start"] is not None:
        current_segment["end"] = start_time + duration
        segments.append(current_segment)
    
    return segments

def format_time(seconds):
    """Format seconds into MM:SS format."""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"

def create_embeddings(text):
    """Create embeddings for a text using OpenAI's API."""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def retrieve_relevant_segments(query, segments, segment_embeddings, top_k=3):
    """Retrieve the top_k most relevant segments for a query."""
    query_embedding = create_embeddings(query)
    
    if query_embedding is None:
        return []
    
    # Calculate similarities
    similarities = [similarity(query_embedding, seg_emb) for seg_emb in segment_embeddings]
    
    # Get indices of top_k most similar segments
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [segments[i] for i in top_indices]

def generate_response(query, relevant_segments):
    """Generate a response based on the query and relevant segments."""
    # Prepare the context from relevant segments
    context = "\n\n".join([
        f"Segment {i+1} (Time: {format_time(seg['start'])} - {format_time(seg['end'])}):\n{seg['text']}"
        for i, seg in enumerate(relevant_segments)
    ])
    
    prompt = f"""
You are an assistant that helps users understand YouTube videos. 
Your task is to answer the user's question based on the provided transcript segments.
Include timestamps in your answer to reference where in the video the information comes from.

USER QUESTION: {query}

RELEVANT TRANSCRIPT SEGMENTS:
{context}

INSTRUCTIONS:
1. Answer the question accurately based on the provided transcript segments.
2. Include specific timestamps from the segments to back up your answer.
3. If you cannot answer from the provided segments, say so clearly.
4. Be concise but informative.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I couldn't generate a response due to an error."

def main():
    st.set_page_config(page_title="YouTube Video Chat", page_icon="🎥", layout="wide")
    
    st.title("🎥 YouTube Video Chat")
    st.write("Ask questions about any YouTube video with an available transcript!")
    
    # Initialize session state variables
    if 'segments' not in st.session_state:
        st.session_state.segments = None
    if 'segment_embeddings' not in st.session_state:
        st.session_state.segment_embeddings = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Video URL Input
    youtube_url = st.text_input("Enter YouTube URL:", key="youtube_url")

    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Load Video"):
            if youtube_url:
                with st.spinner("Loading transcript and preparing embeddings..."):
                    # Extract video ID
                    video_id = extract_video_id(youtube_url)
                    if video_id:
                        st.session_state.video_id = video_id
                        
                        # Get transcript
                        transcript = get_transcript(video_id)
                        if transcript:
                            # Segment transcript
                            segments = segment_transcript(transcript)
                            st.session_state.segments = segments
                            
                            # Create embeddings for each segment
                            segment_embeddings = [create_embeddings(seg["text"]) for seg in segments]
                            st.session_state.segment_embeddings = segment_embeddings
                            
                            st.success(f"Loaded transcript with {len(segments)} segments")
                    else:
                        st.error("Could not extract video ID from the URL")
            else:
                st.warning("Please enter a YouTube URL")
    
    with col2:
        if st.session_state.video_id:
            st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
    
    # Chat interface
    st.header("Ask about the video")
    user_query = st.text_input("Your question:", key="user_query")
    
    if st.button("Ask") and user_query:
        if st.session_state.segments and st.session_state.segment_embeddings:
            with st.spinner("Generating response..."):
                # Retrieve relevant segments
                relevant_segments = retrieve_relevant_segments(
                    user_query, 
                    st.session_state.segments, 
                    st.session_state.segment_embeddings
                )
                
                # Generate response
                response = generate_response(user_query, relevant_segments)
                
                # Add to chat history
                st.session_state.chat_history.append({"query": user_query, "response": response})
        else:
            st.warning("Please load a video first")
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.subheader(f"Q: {chat['query']}")
            st.write(f"A: {chat['response']}")
            st.markdown("---")

if __name__ == "__main__":
    main()
