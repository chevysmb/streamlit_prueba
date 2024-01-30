import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, set_global_service_context
from llama_index.llms import Ollama
from llama_index import SimpleDirectoryReader
from pathlib import Path
from llama_index import download_loader

st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=Ollama(model="mistral", temperature=0.1, system_prompt="You are currently a chatbot with the ability to answer questions about the PDF data files that are passed to you, this includes describing in detail the content of a document that has essays by a person and another document that contains a set of relevant news from the last month, when you answer, do so accurately, mentioning relevant data from the information you have from the pdf."), embed_model='local')
        set_global_service_context(service_context)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
