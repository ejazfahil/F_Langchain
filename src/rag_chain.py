"""RAG chain. 2025-05-02"""
from typing import List

def build_rag_chain(docs:List[str], api_key:str):
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    embeddings=OpenAIEmbeddings(api_key=api_key)
    vs=FAISS.from_documents([Document(page_content=d) for d in docs],embeddings)
    llm=ChatOpenAI(model="gpt-4o-mini",api_key=api_key,temperature=0)
    return RetrievalQA.from_chain_type(llm=llm,retriever=vs.as_retriever(search_kwargs={"k":4}),
                                       return_source_documents=True)
