# Semantic Movie Recommendation Engine

An end-to-end semantic search application powered by vector embeddings, ChromaDB, and Streamlit. Rather than relying on traditional keyword or substring matching, this project leverages transformer-based sentence embeddings to capture semantic meaning—allowing users to search for movies using natural language plot descriptions, themes, or abstract concepts.

**Deployed App:** https://aipi510-week13-movie-search.streamlit.app/

## Architecture & Workflow

The application is designed around a decoupled two-stage pipeline separating offline data ingestion from real-time inference:

1. **Offline Data Ingestion (`load_db.py`)**: 
   * Cleans and preprocesses raw tabular data from the IMDb Top 1000 dataset using Pandas.
   * Encodes movie plot overviews into 384-dimensional dense vector representations using the pre-trained `sentence-transformers/all-MiniLM-L6-v2` model.
   * Persists the embeddings, raw text documents, and unique identifiers to a local ChromaDB vector store (`./chroma_db`).

2. **Real-Time Query Engine & UI (`app.py`)**:
   * Utilizes Streamlit resource caching (`@st.cache_resource`) to load the transformer model and database client once into memory, minimizing query latency.
   * Embeds incoming user queries on the fly into the same shared vector space.
   * Executes a nearest-neighbor similarity search against ChromaDB to retrieve and display the top 5 most conceptually similar films alongside their original descriptions.

## Repository Structure

```text
.
├── Data/
│   └── imdb_top_1000.csv    # Source dataset containing movie titles and plot overviews
├── chroma_db/
│   └── chroma.sqlite3       # Embedding database
├── load_db.py               # ETL and embedding generation pipeline
├── app.py                   # Interactive Streamlit web application
└── README.md                # Project documentation
