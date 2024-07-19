import os
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import PyPDF2
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# app = Flask(__name__)

# folder_path = "db"

# cached_llm = Ollama(model="llama3")

# embedding = FastEmbedEmbeddings()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
# )

# raw_prompt = PromptTemplate.from_template(
#     """ 
#     <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
#     [INST] {input}
#            Context: {context}
#            Answer:
#     [/INST]
# """
# )

# @app.route("/ai", methods=["POST"])
# def aiPost():
#     print("Post /ai called")
#     json_content = request.json
#     query = json_content.get("query")

#     print(f"query: {query}")

#     response = cached_llm.invoke(query)

#     print(response)

#     response_answer = {"answer": response}
#     return response_answer


# @app.route("/ask_pdf", methods=["POST"])
# def askPDFPost():
#     print("Post /ask_pdf called")
#     json_content = request.json
#     query = json_content.get("query")

#     print(f"query: {query}")

#     print("Loading vector store")
#     vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

#     # Debug: Check if vector store contains documents
#     documents = vector_store.get_all_documents()
#     print(f"Documents in vector store: {documents}")

#     print("Creating chain")
#     retriever = vector_store.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={
#             "k": 20,
#             "score_threshold": 0.0,  # Lower threshold for testing
#         },
#     )

#     # Debug: Check retriever functionality
#     print("Testing retriever")
#     test_results = retriever.retrieve(query)
#     print(f"Retriever results: {test_results}")

#     if not test_results:
#         print("No relevant documents found. Consider adjusting search parameters.")

#     document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
#     chain = create_retrieval_chain(retriever, document_chain)

#     result = chain.invoke({"input": query})

#     print(result)

#     sources = []
#     for doc in result["context"]:
#         sources.append(
#             {"source": doc.metadata["source"], "page_content": doc.page_content}
#         )

#     response_answer = {"answer": result["answer"], "sources": sources}
#     return response_answer


# @app.route("/pdf", methods=["POST"])
# def pdfPost():
#     file = request.files["file"]
#     file_name = file.filename
#     save_file = "pdf/" + file_name
#     file.save(save_file)
#     print(f"filename: {file_name}")

#     loader = PDFPlumberLoader(save_file)
#     docs = loader.load_and_split()
#     print(f"docs len={len(docs)}")

#     chunks = text_splitter.split_documents(docs)
#     print(f"chunks len={len(chunks)}")

#     # Debug: Print the content of a few chunks
#     for i, chunk in enumerate(chunks[:5]):
#         print(f"Chunk {i}: {chunk}")

#     vector_store = Chroma.from_documents(
#         documents=chunks, embedding=embedding, persist_directory=folder_path
#     )
#     vector_store.persist()

#     response = {
#         "status": "Successfully Uploaded",
#         "filename": file_name,
#         "doc_len": len(docs),
#         "chunks": len(chunks),
#     }
#     return response


# def start_app():
#     app.run(host="0.0.0.0", port=8080, debug=True)


# if __name__ == "__main__":
#     start_app()


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to add documents to vector store and log them
def add_documents_to_vector_store(documents):
    vector_store = Chroma(embedding_function=OpenAIEmbeddings())
    vector_store.add_documents(documents)
    for doc in documents:
        print(f"Added document: {doc['content'][:500]}")  # Log the first 500 characters of each document
    return vector_store

# Endpoint to upload and process PDF
@app.route('/pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    filepath = os.path.join("pdf", file.filename)
    file.save(filepath)
    
    # Parse the PDF and log content
    documents = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text = page.extractText()
            logging.info(f"Page {page_num}: {text[:500]}")  # Log the first 500 characters of each page
            documents.append({"content": text})
    
    # Add documents to vector store
    vector_store = add_documents_to_vector_store(documents)
    
    return "PDF processed", 200

# Function to retrieve answer from vector store
def get_answer_from_vector_store(query):
    vector_store = Chroma(embedding_function=OpenAIEmbeddings())
    results = vector_store.query(query, top_k=5, threshold=0.05)  # Adjust threshold and top_k as needed
    if results["documents"]:
        answer = results["documents"][0]["content"]
        sources = results["documents"]
    else:
        answer = "No relevant documents found."
        sources = []
    return answer, sources

# Endpoint to handle queries related to the PDF
@app.route('/ask_pdf', methods=['POST'])
def ask_pdf():
    data = request.get_json()
    query = data['query']
    
    # Retrieve answer from vector store
    answer, sources = get_answer_from_vector_store(query)
    return jsonify({'answer': answer, 'sources': sources})

# Additional endpoint to handle other AI-related queries
@app.route('/ai', methods=['POST'])
def ai():
    data = request.get_json()
    query = data['query']
    
    # Process the AI query (this is just a placeholder; replace with your actual AI logic)
    response = {"answer": f"Processed AI query: {query}"}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8080)
