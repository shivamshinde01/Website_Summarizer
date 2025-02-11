import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up Google Gemini API Key
genai.configure(api_key="AIzaSyBTR3MmV7TiGPznME9B5CODfOkyVb2_sa0")

# Initialize Sentence Transformer model for embeddings
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("🌐 AI-Based Website Summarizer & Chatbot")
st.write("Enter a website URL to summarize its content and ask questions about it.")

# Function to extract website content
def extract_website_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        headings = soup.find_all(["h1", "h2", "h3"])
        text = "\n".join([h.get_text() for h in headings]) + "\n" + " ".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# User input for website URL
website_url = st.text_input("🔗 Enter Website URL:")

if st.button("Summarize Website"):
    website_text = extract_website_text(website_url)
    
    if "Error" in website_text:
        st.error("⚠️ Could not fetch website content. Please check the URL.")
    else:
        # Generate a more descriptive summary using AI
        summary_prompt = f"""Extract the following details from this website:
Website Name:
/Short Summary:

{website_text[:4000]}"""
        summary_response = genai.GenerativeModel("gemini-pro").generate_content(summary_prompt)
        
        # Display summary
        st.subheader("📌 Website Summary")
        st.write(summary_response.text if summary_response else "No summary available.")

        # Prepare chatbot by creating embeddings for website text
        text_chunks = website_text.split(". ")
        text_embeddings = model_embed.encode(text_chunks)
        
        # Store embeddings in FAISS index for fast search
        index = faiss.IndexFlatL2(text_embeddings.shape[1])
        index.add(np.array(text_embeddings))
        
        # Store extracted website content in session state for chatbot use
        st.session_state.website_text = text_chunks
        st.session_state.index = index
        st.session_state.model_embed = model_embed
        st.session_state.website_url = website_url

# Chatbot Section
if "website_text" in st.session_state:
    st.subheader("🤖 Chat with Website Content")
    user_query = st.text_input("🔍 Ask a question about the website:")

    if st.button("Get Answer"):
        query_embedding = st.session_state.model_embed.encode([user_query])
        _, result_index = st.session_state.index.search(np.array(query_embedding), 1)
        extracted_answer = st.session_state.website_text[result_index[0][0]]

        refined_prompt = f"Provide a clear, structured, and brand-focused answer based strictly on the website information:\n{extracted_answer}"
        ai_response = genai.GenerativeModel("gemini-pro").generate_content(refined_prompt)

        st.write(f"**🤖 AI Answer:** {ai_response.text if ai_response else extracted_answer}")
