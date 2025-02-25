import re
import openai
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

def get_video_transcript(video_id):
    """Fetches transcript from YouTube video."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript

def process_transcript(transcript):
    """Processes transcript into structured text with timestamps."""
    text = " ".join([entry['text'] for entry in transcript])
    timestamp_map = {idx: entry['start'] for idx, entry in enumerate(transcript)}
    return text, timestamp_map

def split_transcript(text, chunk_size=250, overlap=50):
    """Splits transcript into meaningful chunks while preserving context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def build_vector_store(chunks):
    """Embeds and stores transcript chunks in FAISS for retrieval."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def search_transcript(vector_store, query, top_k=3):
    """Retrieves relevant transcript sections based on user query."""
    results = vector_store.similarity_search(query, k=top_k)
    return results

def generate_response(query, results):
    """Generates a response using GPT based on retrieved transcript sections."""
    context = "\n".join([result.page_content for result in results])
    prompt = f"""
    Based on the following transcript excerpts, answer the question.
    Transcript:
    {context}
    Question: {query}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI assistant that extracts information from YouTube videos."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def main(video_id, user_query):
    transcript = get_video_transcript(video_id)
    text, timestamp_map = process_transcript(transcript)
    chunks = split_transcript(text)
    vector_store = build_vector_store(chunks)
    results = search_transcript(vector_store, user_query)
    response = generate_response(user_query, results)
    return response

if __name__ == "__main__":
    load_dotenv(dotenv_path='.env')
    video_id = "4GLSzuYXh6w"
    user_query = "What is the main topic discussed?"
    print(main(video_id, user_query))
