import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("ticket_collection")

def ingest_tickets(tickets: list) -> None:
    """
    Ingests a list of tickets into the ChromaDB collection.

    Args:
        tickets (list): A list of ticket dictionaries, each containing 'ticket_text' and 'embedding'.
    """
    for ticket in tickets:
        file_name = ticket["file_name"]
        chunk_index = ticket["chunk_index"]
        ticket_id = f"{file_name}_chunk_{chunk_index}"
        collection.add(
            documents=[ticket['text']],
            embeddings=[ticket['embedding']],
            ids= [ticket_id]
        )

