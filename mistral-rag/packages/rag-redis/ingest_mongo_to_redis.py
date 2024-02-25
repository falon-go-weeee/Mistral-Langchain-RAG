import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from rag_redis.json_loader import JSONLoader
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.redis import Redis
from rag_redis.config import EMBED_MODEL, INDEX_NAME, INDEX_SCHEMA, REDIS_URL


def ingest_documents():
    """
    Ingest PDF to Redis from the data/ directory that
    contains Edgar 10k filings data for Nike.
    """
    # Load list of pdfs
    company_name = "Nike"
    data_path = "data/"
    doc = [os.path.join(data_path, file) for file in os.listdir(data_path)][0]

    print("Parsing 10k filing doc for NIKE", doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, add_start_index=True
    )
    loader = UnstructuredFileLoader(doc, mode="single", strategy="fast")
    chunks = loader.load_and_split(text_splitter)

    print("Done preprocessing. Created", len(chunks), "chunks of the original pdf")
    # Create vectorstore
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    _ = Redis.from_texts(
        # appending this little bit can sometimes help with semantic retrieval
        # especially with multiple companies
        texts=[f"Company: {company_name}. " + chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embedding=embedder,
        index_name=INDEX_NAME,
        index_schema=INDEX_SCHEMA,
        redis_url='',
    )

def ingest_mc_data():
    """
    Ingest JSON's to Redis from the data/ directory
    """
    
    data_path = "data/2023-12-26/"
    json_path_list = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.json')]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, add_start_index=True
    )
    
    data_ls = []
    metadata_ls = []
    for json_file_path in json_path_list:
        with open(json_file_path) as f:
            json_parse = json.load(f)
        # article_content = json_parse['article']
        # title = json_parse['title']
        
        article_content = JSONLoader(
                            file_path=json_file_path,
                            content_key='article',
                        )
        title = JSONLoader(
                            file_path=json_file_path,
                            content_key='title',
                        )
        print(title.load()[0].page_content)
        chunks = article_content.load_and_split(text_splitter)
        print(chunks)
        data_ls.extend([f"article_title: {title.load()[0].page_content}. " + chunk.page_content for chunk in chunks])
        metadata_ls.extend([chunk.metadata for chunk in chunks])
        if len(chunks)>0:
            break
    # print(data_ls)
    print("Done preprocessing. Created", len(data_ls), "chunks of JSON data")
    # Create vectorstore
    # embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # _ = Redis.from_texts(
    #     # appending this little bit can sometimes help with semantic retrieval
    #     # especially with multiple companies
    #     texts=data_ls,
    #     metadatas=metadata_ls,
    #     embedding=embedder,
    #     index_name=INDEX_NAME,
    #     index_schema=INDEX_SCHEMA,
    #     redis_url=REDIS_URL,
    # )

def mongo_to_redis():
    """
    Ingest MongoDB to Redis from the data/ directory
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, add_start_index=True
    )
    loader = MongodbLoader(
        connection_string="",
        db_name="",
        collection_name="",
    )
    chunks = loader.load_and_split(text_splitter)
    print(chunks)
    print("Done preprocessing. Created", len(chunks), "chunks of the original pdf")
    # Create vectorstore
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    _ = Redis.from_texts(
        # appending this little bit can sometimes help with semantic retrieval
        # especially with multiple companies
        texts=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embedding=embedder,
        index_name=INDEX_NAME,
        index_schema=INDEX_SCHEMA,
        redis_url='',
    )
    
if __name__ == "__main__":
    mongo_to_redis()
