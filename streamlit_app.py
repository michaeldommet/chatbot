import streamlit as st
import vertexai
from vertexai.preview.language_models import ChatModel


PROJECT_ID = "root-fort-398913"
vertexai.init(project=PROJECT_ID, location="us-central1")

# App title
st.set_page_config(page_title="💬 ChatBot")

# Hugging Face Credentials
with st.sidebar:
    st.title('💬 ChatBot')
    model_option = st.selectbox('Chose the Model you want to work with',
                                ('chat-bison@001', 'text-bison@001', 'chat-bison'))

    st.write('You selected:', model_option)
    st.markdown(
        '📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response


def generate_response(prompt):
    """Predict using a Large Language Model."""
    model = ChatModel.from_pretrained(model_option)
    chat = model.start_chat(examples=[])
    response = chat.send_message(prompt)
    output = response.text
    return output


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
