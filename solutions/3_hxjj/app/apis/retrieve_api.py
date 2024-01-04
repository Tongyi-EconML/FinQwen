from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from retrieve.doc_retrieve import DocRetrieve

def embedding_model():
    model_path = 'models/text2vec-bge-large-chinese'
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_path,model_kwargs={'device': 'cuda'})
    return embeddings

doc_retrieve = DocRetrieve(20)

