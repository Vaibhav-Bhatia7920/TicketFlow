from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path

def chunk_text(directory: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Splits the input text into chunks of specified size with overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")
    chunks_list = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(type(text))
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks_list.append({"chunks":splitter.split_text(text),"file_name":file})
    
    return chunks_list