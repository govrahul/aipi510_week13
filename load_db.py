import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("movie_collection")

df = pd.read_csv("Data/imdb_top_1000.csv", usecols=['Series_Title', 'Overview'])
df.dropna(subset=['Series_Title', 'Overview'])
df.drop_duplicates(subset=['Series_Title'])

unique_ids = [f"{title}_{idx}" for idx, title in enumerate(df['Series_Title'])]

embeddings = model.encode(df['Overview'].tolist())

collection.add(
               documents=df['Overview'].tolist(), 
               embeddings=embeddings.tolist(), 
               ids=unique_ids
               )

print(f"Successfully loaded {len(df)} records to database")