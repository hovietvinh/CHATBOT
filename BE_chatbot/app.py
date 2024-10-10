import os
import re
import warnings
from flask import Flask, request, jsonify, make_response
from dotenv import load_dotenv
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnableSequence
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypdfium2")

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = "gemini-pro"
PDF_DIRECTORY = "./resources"
PERSIST_DIRECTORY = "./chroma_db"
SIMILARITY_THRESHOLD = 0.3

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(google_api_key=API_KEY, model=GEMINI_MODEL)
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2')
vectorizer = TfidfVectorizer()

def load_pdfs_with_keywords(directory_path):
    all_documents = []
    all_text = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            loader = PyPDFium2Loader(pdf_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = filename
                all_documents.append(doc)
                all_text.append(doc.page_content)

    if not all_documents or not all_text:
        raise ValueError("No documents or text found in the specified directory.")
    
    tfidf_matrix = vectorizer.fit_transform(all_text)
    return all_documents, tfidf_matrix

def filter_documents_by_source(documents, source_file):
    return [doc for doc in documents if doc.metadata.get("source_file") == source_file]

def determine_source_file(question, documents, tfidf_matrix):
    question_tfidf = vectorizer.transform([question])
    similarities = (tfidf_matrix * question_tfidf.T).toarray().flatten()
    print(f"Question: {question}")
    print(f"TF-IDF similarities: {similarities}")
    document_scores = [(doc, sim) for doc, sim in zip(documents, similarities) if sim > 0]
    if document_scores:
        document_scores.sort(key=lambda x: x[1], reverse=True)
        top_document = document_scores[0]
        return top_document[0].metadata.get("source_file")
    
    return None

# Load PDFs from the directory
all_documents, tfidf_matrix = load_pdfs_with_keywords(PDF_DIRECTORY)

# Custom Text Splitting
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=200)
split_docs = []

def split_document(doc):
    content = doc.page_content
    topic_pattern = re.compile(r"^Topic\s+\d+:\s+(.+)$", re.MULTILINE)
    section_pattern = re.compile(r"^\d+\.\s+(.+)$", re.MULTILINE)
    bullet_pattern = re.compile(r"^\s*•\s+(.+)$", re.MULTILINE)
    topics = topic_pattern.split(content)
    
    if len(topics) > 1:
        for i in range(1, len(topics), 2):
            topic_title = topics[i].strip()
            topic_content = topics[i + 1].strip()

            sections = section_pattern.split(topic_content)
            if len(sections) > 1:
                for j in range(1, len(sections), 2):
                    section_title = sections[j].strip()
                    section_content = sections[j + 1].strip()

                    bullets = bullet_pattern.split(section_content)
                    if len(bullets) > 1:
                        for k in range(1, len(bullets), 2):
                            bullet_content = bullets[k].strip()
                            metadata = {
                                "source_file": doc.metadata.get("source_file"),
                                "topic": topic_title,
                                "section": section_title,
                                "bullet_point": bullet_content[:30]  # First 30 characters of the bullet point
                            }
                            split_docs.append(Document(page_content=bullet_content, metadata=metadata))
                    else:
                        metadata = {
                            "source_file": doc.metadata.get("source_file"),
                            "topic": topic_title,
                            "section": section_title,
                        }
                        chunks = text_splitter.split_text(section_content)
                        for chunk in chunks:
                            split_docs.append(Document(page_content=chunk, metadata=metadata))
            else:
                metadata = {
                    "source_file": doc.metadata.get("source_file"),
                    "topic": topic_title,
                }
                chunks = text_splitter.split_text(topic_content)
                for chunk in chunks:
                    split_docs.append(Document(page_content=chunk, metadata=metadata))
    else:
        metadata = {
            "source_file": doc.metadata.get("source_file"),
            "topic": "General Information",
        }
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata=metadata))

for doc in all_documents:
    split_document(doc)

# Create and Save Vector Store and compressed to LLMChain
try:
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
except Exception as e:
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_kwargs={"k": 5})
)

template = """You are a friendly and enthusiastic AI chatbot named Niko specializing in Human-Computer Interaction (HCI) in Hanoi University. Your role is to help students like me understand the subject. Think of yourself as my personal tutor.

If I don't ask a question directly related to HCI or don't found the relevant course, feel free to introduce yourself or chat with me!

When I do ask about HCI, please use the following context from the course material from the school to answer my questions as thoroughly as possible. If you don't find enough information to provide a complete answer, mention what information is missing and encourage me to ask follow-up questions or seek additional resources. Always maintain a positive and encouraging tone to support my learning.

Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
runnable_sequence = RunnableSequence(prompt | llm)


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    user_message = request.json.get('message')
    answer = "I'm having trouble processing that right now. Please try again later."
    try:       
        generic_responses = ["hello", "hi", "hey", "greetings"]
        if any(greet in user_message.lower() for greet in generic_responses):
            return jsonify({'response': "Hi! I'm Niko, your friendly HCI tutor. How can I assist you with Human-Computer Interaction today?"})
        source_file = determine_source_file(user_message, all_documents, tfidf_matrix)
        if source_file:
            print(f"Determined source file: {source_file}")
            retrieved_docs = vectordb.similarity_search(
                user_message, 
                k=5, 
                filter={"source_file": source_file}
            )
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            compressed_docs = retriever.invoke(user_message)
            compressed_context = "\n\n".join([doc.page_content for doc in compressed_docs])
            context_to_use = compressed_context if compressed_context else context
            
            response = runnable_sequence.invoke({"context": context_to_use, "question": user_message})
            
            if len(response.content) < 100:
                try:
                    bm25_retriever = BM25Retriever.from_documents(retrieved_docs)
                    bm25_docs = bm25_retriever.invoke(user_message)
                    all_retrieved_docs = retrieved_docs + bm25_docs
                    context = "\n\n".join([doc.page_content for doc in all_retrieved_docs])
                    response = runnable_sequence.invoke({"context": context, "question": user_message})
                except Exception as e:
                    print(f"Error during BM25 retrieval: {e}")
            
            answer = response.content if response else "I don't have enough information to answer that question about HCI."
        else:
            answer = "I couldn't find relevant course material for your question. Could you provide more details or ask something specific about HCI?"
    except Exception as e:
        print(f"Error processing request: {e}")
        answer = "I'm having trouble processing that right now. Please try again later."
    return jsonify({'response': answer.replace('\n', '<br>').strip()})


if __name__ == '__main__':
    app.run(debug=True, port=5000)