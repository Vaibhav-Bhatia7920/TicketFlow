import chromadb
from app.chroma_db_flow.embedder import query_embedder

def retreive_chunks(query: str) -> list:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("ticket_collection")

    query_embedding = query_embedder(query)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    print(result["documents"])
    return result["documents"]  # Return the list of retrieved chunks