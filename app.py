import streamlit as st
import pandas as pd
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

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

Then, your output should have the following: 
An Area of Focus and a description of that area of focus
Then underneath the area of focus there should be different initiatives and an action step for initiative
Have one focus area three initiatives and two action steps for each initiative

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

# Function to generate fake data
def generate_fake_data():
    # Create date range for two years per month
    dates = pd.date_range(end=pd.Timestamp.today(), periods=24, freq='M')

    # Initialize DataFrame
    data = pd.DataFrame({'Date': np.tile(dates, 3)})

    # Metrics
    metrics = ['Denials Rate', 'ER Visits', 'In-Patient Length of Stay']
    data['Metric'] = np.repeat(metrics, len(dates))

    # Generate data for each metric
    values = []
    increasing = []

    for metric in metrics:
        if metric == 'ER Visits':
            # Generate increasing data with randomness
            start = 100
            end = 200
            trend = np.linspace(start, end, len(dates))
            noise = np.random.normal(0, 5, len(dates))  # Adding randomness
            trend += noise
            values.extend(trend)
            increasing.extend(['Yes'] * len(dates))
        else:
            # Generate random data without increasing trend
            mean = 50
            std_dev = 5
            random_data = np.random.normal(mean, std_dev, len(dates))
            values.extend(random_data)
            increasing.extend(['No'] * len(dates))

    data['Value'] = values
    data['Increasing'] = increasing

    return data

# Streamlit application
st.title("RCM Decision Support Assistant")
st.write("Ask me how I can help improve your Revenue Cycle Management (RCM) processes.")

# Generate and display the fake data
data = generate_fake_data()

st.header("Metrics Over Time")
metrics = data['Metric'].unique()
# Set "ER Visits" as the default selection
# Convert default_index to int to avoid StreamlitAPIException
default_index = int(np.where(metrics == 'ER Visits')[0][0])
metric_selected = st.selectbox("Select a metric to view its trend:", metrics, index=default_index)

metric_data = data[data['Metric'] == metric_selected]

st.line_chart(metric_data.set_index('Date')['Value'])

st.write("Data Table:")
st.write(metric_data[['Date', 'Metric', 'Value', 'Increasing']])

# Identify the metric where the trend is increasing
increasing_metric = data[data['Increasing'] == 'Yes']['Metric'].unique()

if len(increasing_metric) > 0:
    metric_with_increase = increasing_metric[0]  # Assuming only one metric is increasing
else:
    metric_with_increase = None

# Modify the prompt to include the metric name where the trend is increasing
if metric_with_increase:
    default_question = f"The {metric_with_increase} metric is showing an increasing trend over the past two years. How can we address this?"
else:
    default_question = "No metrics are showing an increasing trend."

st.subheader("Automated Playbook Generation")
st.write(f"Metric with increasing trend: **{metric_with_increase if metric_with_increase else 'None'}**")

if metric_with_increase:
    # Use the metric name in the question to generate the playbook
    generated_response = qa_chain.run(default_question)
    st.write(generated_response)
else:
    st.write("No increasing trends detected.")

# Allow the user to enter their own question
st.subheader("Your Custom Playbook")
user_input = st.text_input("Enter your question or issue:")

use_rag = st.checkbox("Use Retrieval-Augmented Generation (RAG)", value=True)

if user_input:
    with st.spinner("Generating response..."):
        if use_rag:
            response = qa_chain.run(user_input)
        else:
            response = generate_response_without_rag(user_input)
    st.write(response)
