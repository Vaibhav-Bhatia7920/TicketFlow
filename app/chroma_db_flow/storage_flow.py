from app.chroma_db_flow.chunker import chunk_text
from app.chroma_db_flow.embedder import embed_text
from app.chroma_db_flow.ingester import ingest_tickets


def process_and_ingest_ticket(ticket_text: str):
    """
    Processes the ticket text by chunking, embedding, and ingesting into ChromaDB.

    Args:
        ticket_text (str): The text of the ticket to be processed.
    """
    # Step 1: Chunk the ticket text
    chunks = chunk_text(ticket_text)

    # Step 2: Generate embeddings for each chunk
    embeddings = embed_text(chunks)


    # Step 4: Ingest the tickets into ChromaDB
    ingest_tickets(embeddings)

if __name__ == "__main__":
    ticket_text = "This is a sample ticket text for testing purposes."
    path = "/Users/vaibhav/Documents/DevBase/Ticket Resolver/Dev/app/ticket_dataset/"
    process_and_ingest_ticket(path)