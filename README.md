

# RCM Decision Support Assistant
The Mede Decision Support Assistant is a Streamlit application designed to help users improve their Revenue Cycle Management (RCM) processes. It leverages GPT-4 and Retrieval-Augmented Generation (RAG) to provide context-rich responses based on a custom RCM knowledge base.

# Features
Retrieval-Augmented Generation (RAG): Enhances responses by integrating information from your RCM knowledge base.
GPT-4 Powered: Utilizes OpenAI's GPT-4 model for generating insightful answers.
Interactive Interface: Simple and user-friendly interface built with Streamlit.
Customizable Knowledge Base: Easily replace or expand the RCM content with your own data.

# Prerequisites
Python 3.8 or higher
OpenAI API Key: Access to GPT-4 is required.
Required Python Libraries:
streamlit
langchain
openai
tiktoken
faiss-cpu (or faiss-gpu if using GPU acceleration)
pydantic (version less than 2)
