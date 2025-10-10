"""Conversation with memory. 2025-10-10"""
def build_chat_chain(api_key:str, use_summary:bool=True):
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    llm=ChatOpenAI(model="gpt-4o-mini",api_key=api_key,temperature=0.7)
    mem=(ConversationSummaryMemory(llm=llm) if use_summary else ConversationBufferMemory())
    return ConversationChain(llm=llm,memory=mem,verbose=False)
