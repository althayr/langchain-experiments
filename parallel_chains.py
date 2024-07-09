import sys
import time
import logging
from typing import List
from operator import itemgetter
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class ToyRetriever(BaseRetriever):
    """Custom retriever adapted from LangChain Retriever documentation, it stores documents in memory and returns the top k matches for each query.
    """

    documents: List[Document]
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        start = time.time()
        logger.info(f"Processing query: '{query}'")
        
        # Simulating a 3 seconds delay for the response
        time.sleep(3)
        
        matching_documents = []
        for document in self.documents:
            if len(matching_documents) > self.k:
                logger.info(f"Replying with results. Time elapsed: {time.time() - start}")
                return matching_documents

            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        logger.info(f"Replying with results. Time elapsed: {time.time() - start}")
        return matching_documents


if __name__ == "__main__":
    # Adapted from Custom Retriever documentation: https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/custom_retriever/
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"type": "dog", "trait": "loyalty"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"type": "cat", "trait": "independence"},
        ),
        Document(
            page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
            metadata={"type": "fish", "trait": "low maintenance"},
        ),
        Document(
            page_content="Parrots are intelligent birds capable of mimicking human speech.",
            metadata={"type": "bird", "trait": "intelligence"},
        ),
        Document(
            page_content="Rabbits are social animals that need plenty of space to hop around.",
            metadata={"type": "rabbit", "trait": "social"},
        ),
    ]
    retriever = ToyRetriever(documents=documents, k=3)

    # Single thread mode
    logger.info("Running single worker mode - Simple request")
    start = time.time()
    chain = RunnablePassthrough() | retriever
    chain.invoke("that")
    logger.info(f"Done. Time elapsed: {time.time() - start}")
    
    # Sequential mode
    logger.info("Running sequential mode - 5 requests")
    start = time.time()
    for i in range(5):
        chain.invoke('that')
    logger.info(f"Done. Time elapsed: {time.time() - start}")

    # Parallel mode
    logger.info("Running parallel mode - 5 requests")
    start = time.time()
    retrievers_map = {str(i): ToyRetriever(documents=documents, k=3) for i in range(5)}
    map_chain = RunnableParallel(**retrievers_map)
    map_chain.invoke('that')
    logger.info(f"Done. Time elapsed: {time.time() - start}")
