import random
import os
import logging
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from chromadb.config import Settings

# Disable Chroma telemetry and other analytical trackers
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_NO_PROXY"] = "True"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "True"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logger = logging.getLogger("interview")

load_dotenv()

class ChromaDB:
    """
    A wrapper class for managing a Chroma vector database using LangChain.

    This class provides methods for storing, retrieving, and deleting documents 
    in a vector database while utilizing local Ollama embeddings.
    """
    def __init__(self, session_id: str):
        """
        Initializes the ChromaDB instance with a user-specific collection.

        - Sets up a persistent directory for storage.
        - Uses local Ollama embeddings for vector representation.
        - Maintains a set of used document IDs to prevent duplicate retrievals.

        """
        # Ensure alphanumeric and underscores only, and valid length for ChromaDB
        sanitized_session = "".join(c for c in session_id if c.isalnum() or c == "_")
        if not sanitized_session:
            import uuid
            sanitized_session = str(uuid.uuid4()).replace("-", "")
        self.collection_name = f"user_{sanitized_session}"
        self.persist_directory = "./chroma_langchain_db"  # persistent storage
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=OllamaEmbeddings(
                                    model="twine/mxbai-embed-xsmall-v1"
                                ),
            persist_directory=self.persist_directory,
            client_settings=Settings(anonymized_telemetry=False)
        )
        self.used_ids = set()  # store used document ID

    def get_all_documents(self) -> list[dict]:

        """
        Retrieves all stored documents from the Chroma vector database.

        Returns:
            list[dict]: A list of dictionaries, each containing:
                - "id" (str): The document ID.
                - "content" (str): The stored document content.
                - "metadata" (dict): The associated metadata.
        """

        collection = self.vectorstore.get()  # get stored documents
        documents = collection.get("documents",[])  
        ids = collection.get("ids",[])  # extract document IDs
        metadatas = collection.get("metadatas", [{} for _ in range(len(documents))])
        logger.info(f"Successfully retrieved all {len(documents)} documents.")
        return [{"id": doc_id, "content": doc, "metadata": meta} for doc, doc_id, meta in zip(documents, ids, metadatas)]

    def query_vdb(self, query: str, k: int = 1) -> list[dict]:
        """
        Performs a similarity search on the vector database using the provided query string.
        
        Args:
            query (str): The search query (e.g., the current interview question).
            k (int): The number of relevant documents to retrieve.
            
        Returns:
            list[dict]: A list of relevant documents with content and metadata.
        """
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Similarity search for query '{query[:50]}...' returned {len(results)} results.")
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_random_document(self)-> dict | None:

        """
        Retrieves a random document from the stored documents without repetition.

        - Ensures that each document is returned only once until all are used.
        - Resets when all documents have been retrieved.

        Returns:
            dict | None: A dictionary containing "content" and "metadata", 
                        or None if no available documents remain.
        """
        
        all_docs = self.get_all_documents()  
        available_docs = [doc for doc in all_docs if doc["id"] not in self.used_ids]

        if not available_docs:
            return None

        selected_doc = random.choice(available_docs)  # select one at random
        self.used_ids.add(selected_doc["id"])  # mark as used
        logger.info(f"Random document with ID {selected_doc['id']} extracted") 
        return {"content": selected_doc["content"], "metadata": selected_doc["metadata"]}

    def insert_into_chroma(self, extracted_text: str, metadata: dict = None) -> None:
        """
        Splits the given text into smaller chunks and inserts them into the vector database with metadata.
        """
        import time
        start_time = time.perf_counter()
        try:
            text_split = RecursiveCharacterTextSplitter(separators=["\n\n","\n",". "," ",""],
                                                chunk_size=500, chunk_overlap=50)
            documents = text_split.split_text(extracted_text)
            
            # Use random suffix for IDs to ensure uniqueness across multiple calls
            import uuid
            batch_id = str(uuid.uuid4())[:8]
            ids = [f"id_{batch_id}_{i}" for i in range(len(documents))]
            metadatas = [metadata or {} for _ in range(len(documents))]
            
            logger.info(f"Adding {len(documents)} text chunks to Chroma...")
            self.vectorstore.add_texts(texts=documents, ids=ids, metadatas=metadatas)
            duration = time.perf_counter() - start_time
            logger.info(f"Document insertion into Chroma successful in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Document insertion into Chroma failed after {time.perf_counter() - start_time:.2f}s: {e}")

    def delete_inserted_docs(self)-> None:
        """
        Deletes all stored documents from the Chroma vector database.
        """
        try:
            all_docs = self.vectorstore.get()
            all_ids = all_docs["ids"]

            if all_ids:
                self.vectorstore.delete(all_ids)
            
            self.used_ids = set() # Reset used IDs when clearing DB
            logger.info(f"Successfully cleared previous document from VDB and reset used_ids")
        except Exception as e:
            logger.error(f"Clearing previous document from VDB failed : {e}")

# Global instance removed for multi-user safety