import streamlit as st
import utils
from streaming import StreamHandler
from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# App title
st.set_page_config(page_title="💬 ChatBot")

# Hugging Face Credentials
with st.sidebar:
    st.title('💬 ChatBot')
    model_option = st.selectbox('Chose the Model you want to work with',
                                ('chat-bison@001', 'text-bison@001', 'chat-bison'))

    st.write('You selected:', model_option)
    st.markdown(
        '📖 Learn how to build this app in this [Github]()!')


class ContextChatbot:
    @st.cache_resource
    def setup_chain(_self):
        memory = ConversationBufferMemory()
        llm = ChatVertexAI(model_name=model_option, streaming=True)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        return chain

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})


if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
