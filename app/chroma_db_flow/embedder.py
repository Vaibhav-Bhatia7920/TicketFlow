import ollama

def embed_text(chunks: list) -> list:
    """
    Generates embeddings for the given text using the Ollama API.

    Args:
        text (str): The input text to be embedded.
    Returns:    
        list: A list of embeddings for the input text.
    """
    text_and_embeddings = []
    model = "nomic-embed-text:latest"
    for chunk_file in chunks:
        chunks,file_name = chunk_file
        for i in range(len(chunks)):
            chunk = chunks[i]
            response = ollama.embeddings(model=model, prompt = chunk)
            embedding = response['embedding']
            text_and_embeddings.append({"text": chunk, "embedding" : embedding, "file_name": file_name,"chunk_index": i})
        
    return text_and_embeddings