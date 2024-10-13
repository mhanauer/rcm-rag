import streamlit as st
from langchain.vectorstores import FAISS  # Adjust if necessary
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage  # For message types

# Load environment variables
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI Chat Model
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4")

# Sample RCM knowledge base (you can replace this with your actual data)
rcm_text = """
**Revenue Cycle Management (RCM)** is the financial process used by healthcare systems to track patient care episodes from registration and appointment scheduling to the final payment of a balance. RCM unifies the business and clinical sides of healthcare by coupling administrative data with the treatment a patient receives.

**Key Components of RCM:**
- **Patient Scheduling and Registration**: Collecting patient information and scheduling appointments.
- **Insurance Eligibility Verification**: Confirming coverage to reduce claim denials.
- **Charge Capture**: Recording services provided for billing.
- **Claim Submission**: Sending claims to insurers for reimbursement.
- **Payment Posting**: Recording payments received.
- **Denial Management**: Handling rejected claims to recover revenue.
- **Reporting and Analytics**: Using data to improve financial performance.

**Benefits of Effective RCM:**
- Improved cash flow.
- Reduced administrative costs.
- Enhanced patient satisfaction.
- Increased compliance with regulations.
- Better decision-making through analytics.

**Challenges in RCM:**
- Keeping up with changing regulations.
- Managing denials and rejections.
- Integrating disparate systems.
- Ensuring data security and privacy.
"""

# Split the text into manageable chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
texts = text_splitter.split_text(rcm_text)

# Create embeddings for the texts
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
docsearch = FAISS.from_texts(texts, embeddings)

# Create a retriever
retriever = docsearch.as_retriever()

# Define the custom prompt template
prompt_template = """
You are an expert assistant helping revenue cycle management customers make data-driven decisions.

First, provide a brief summary of the relevant information based on the context provided.

Then, offer up to five specific tips to address the user's question.

Use the following context to inform your answer:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create the RetrievalQA chain with the custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
)

# Function to generate response without RAG
def generate_response_without_rag(question):
    messages = [
        SystemMessage(
            content="You are an expert assistant helping revenue cycle management customers make data-driven decisions. First, provide a brief summary of the relevant information. Then, offer up to five specific tips to address the user's question."
        ),
        HumanMessage(content=question)
    ]
    response = llm(messages)
    return response.content.strip()

# Streamlit application
st.title("Mede Decision Support Assistant")
st.write("Ask me how I can help improve your Revenue Cycle Management (RCM) processes.")

user_input = st.text_input("Your question:")
use_rag = st.checkbox("Use Retrieval-Augmented Generation (RAG)", value=True)

if user_input:
    with st.spinner("Generating response..."):
        if use_rag:
            response = qa_chain.run(user_input)
        else:
            response = generate_response_without_rag(user_input)
    st.write(response)
