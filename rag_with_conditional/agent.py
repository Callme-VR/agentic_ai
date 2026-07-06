# 1-># problem statement:
    
    
    
    
# So lets say there are 3 branches in our college BCA , BBA and B COM 

# A student will choose one of these 3 options, and then a chatbot will be activated. You can ask any question to that chatbot, but the LLM you are using in that chatbot does not have knowledge of the college programme. For that, you will have 2 PDFs  the first one will be for academics, and the second one will be for fee-related things. So technically, we will have 3 conditional paths:

# Answering the question using the academic PDF (using RAG)
# Answering the question using the fee PDF (using RAG)
# Answering general questions based on the LLM's own knowledge
# And every conditional path's response converges to a single node.