import random
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import logging
logger = logging.getLogger("interview")
load_dotenv()

class ChromaDB:
    """
    A wrapper class for managing a Chroma vector database using LangChain.

    This class provides methods for storing, retrieving, and deleting documents 
    in a vector database while utilizing Google's Generative AI embeddings.
    """
    def __init__(self):
        """
        Initializes the ChromaDB instance with a predefined collection.

        - Sets up a persistent directory for storage.
        - Uses Generative AI embeddings for vector representation.
        - Maintains a set of used document IDs to prevent duplicate retrievals.

        """
        self.collection_name = "example_collection"
        self.persist_directory = "./chroma_langchain_db"  # persistent storage
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=GoogleGenerativeAIEmbeddings(
                                    model="models/text-embedding-004",
                                    google_api_key=os.getenv("APIKEY")
                                ),
            persist_directory=self.persist_directory 
        )
        self.used_ids = set()  # store used document ID

    def get_all_documents(self) -> list[dict]:

        """
        Retrieves all stored documents from the Chroma vector database.

        Returns:
            list[dict]: A list of dictionaries, each containing:
                - "id" (str): The document ID.
                - "content" (str): The stored document content.
        """

        collection = self.vectorstore.get()  # get stored documents
        documents = collection.get("documents",[])  
        ids = collection.get("ids",[])  # extract document IDs
        logger.info(f"Successfully retrieved all {len(documents)} documents.")
        return [{"id": doc_id, "content": doc} for doc, doc_id in zip(documents, ids)]

    def get_random_document(self)-> str:

        """
        Retrieves a random document from the stored documents without repetition.

        - Ensures that each document is returned only once until all are used.
        - Resets when all documents have been retrieved.

        Returns:
            str | None: The content of a randomly selected document, 
                        or None if no available documents remain.
        """
        
        all_docs = self.get_all_documents()  
        available_docs = [doc for doc in all_docs if doc["id"] not in self.used_ids]

        if not available_docs:
            return None

        selected_doc = random.choice(available_docs)  # select one at random
        self.used_ids.add(selected_doc["id"])  # mark as used
        logger.info(f"Random document with ID {selected_doc["id"]} extracted") 
        return selected_doc["content"]

    def insert_into_chroma(self,extracted_text: str) -> None:
        """
        Splits the given text into smaller chunks and inserts them into the vector database.

        - Uses a 'RecursiveCharacterTextSplitter' to break text into smaller chunks.
        - Assigns unique IDs to each chunk before storing them.

        Args:
            extracted_text (str): The text content to be stored in the database.
        """
        try:
            text_split = RecursiveCharacterTextSplitter(separators=["\n\n","\n",". "," ",""],
                                                chunk_size=500, chunk_overlap=50)
            documents = text_split.split_text(extracted_text)
            ids = [f"id_{i}" for i in range(len(documents))]
            
            self.vectorstore.add_texts(texts=documents, ids=ids)
            logger.info(f"Document insertion into Chroma successful")
        except Exception as e:
            logger.error(f"Document insertion into Chroma failed : {e}")

    def delete_inserted_docs(self)-> None:
        """
        Deletes all stored documents from the Chroma vector database.

        - Retrieves all document IDs from the database.
        - Removes all documents associated with those IDs.
        """
        try:
            all_docs = self.vectorstore.get()
            all_ids = all_docs["ids"]

            if all_ids:
                self.vectorstore.delete(all_ids)
            logger.info(f"Successfully cleared previous document from VDB")
        except Exception as e:
            logger.error(f"Clearing previous document from VDB failed : {e}")