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
    print(f"Generating embeddings for {len(chunks)} chunks using model '{model}'")
    for chunk_file in chunks:
     
        chunks,file_name = chunk_file['chunks'],chunk_file['file_name']
   
        for i in range(len(chunks)):
            chunk = chunks[i]
            
            response = ollama.embeddings(model=model, prompt = chunk)
            embedding = response['embedding']
            text_and_embeddings.append({"text": chunk, "embedding" : embedding, "file_name": file_name,"chunk_index": i})
        
    return text_and_embeddings

def query_embedder(query: str) -> list:

    model = "nomic-embed-text:latest"
    response = ollama.embeddings(model=model, prompt = query)
    embedding = response['embedding']
    return embedding