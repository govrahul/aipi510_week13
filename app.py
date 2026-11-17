import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Semantic Movie Finder",
    page_icon="🎬",
    layout="centered"
)

# Cache the model loading so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Cache the ChromaDB client connection
@st.cache_resource
def load_chroma_client():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(name="movie_collection")

# Load resources
with st.spinner("Loading embedding model and connecting to database..."):
    model = load_model()
    collection = load_chroma_client()

# App UI Header
st.title("🎬 Semantic Movie Finder")
st.markdown("Type a plot description, theme, or vibe (e.g., *'a heist movie involving dreams'*), and find the top 5 matching movies from your database.")

# User input text box
user_query = st.text_area(
    "Describe a movie:", 
    placeholder="e.g., An epic space adventure with robots and rebellion...",
    height=100
)

# Search execution on button click or Enter key
if st.button("Search Movies", type="primary") or user_query:
    if not user_query.strip():
        st.warning("Please enter a description to search.")
    else:
        with st.spinner("Searching vector database..."):
            # Generate embedding for user query
            query_vector = model.encode(user_query).tolist()
            
            # Query ChromaDB for top 5 results
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=5
            )
            
            titles = results['ids'][0]
            overviews = results['documents'][0]
            
        st.subheader("Top 5 Matching Movies")
        st.divider()
        
        # Display results nicely
        for i, (title, overview) in enumerate(zip(titles, overviews), 1):
            # Clean up the ID suffix (e.g., "Drishyam_42" -> "Drishyam")
            clean_title = title.rsplit('_', 1)[0] if '_' in title else title
            
            st.markdown(f"### {i}. {clean_title}")
            st.write(overview)
            st.markdown("---")