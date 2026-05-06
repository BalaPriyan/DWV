import chromadb
import logging
import os
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryEngine:
    def __init__(self, db_path="memory/chroma_db"):
        if not os.path.exists(os.path.dirname(db_path)):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
        self.client = chromadb.PersistentClient(path=db_path)
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="dwv_memories",
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        logger.info(f"MemoryEngine initialized with storage at {db_path}")

    def store(self, user_input, agent_response):
        """Stores a conversation turn into the vector database."""
        timestamp = datetime.now().isoformat()
        content = f"User: {user_input}\nAgent: {agent_response}"
        
        try:
            self.collection.add(
                documents=[content],
                metadatas=[{"timestamp": timestamp, "user_input": user_input, "agent_response": agent_response}],
                ids=[str(uuid.uuid4())]
            )
            logger.debug(f"Stored new memory: {user_input[:50]}...")
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

    def retrieve(self, query, top_k=3):
        """Retrieves relevant past memories based on the query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if not results or not results['documents'] or not results['documents'][0]:
                return ""
            
            memories = results['documents'][0]
            formatted_memories = "\n---\n".join(memories)
            return formatted_memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return ""
