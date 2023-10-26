import streamlit as st
import utils
import os
from streaming import StreamHandler
from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.utilities.jira import JiraAPIWrapper
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


# Set up Jira authentication
os.environ["JIRA_API_TOKEN"] = "YOUR_JIRA_API_TOKEN"
os.environ["JIRA_USERNAME"] = "YOUR_JIRA_USERNAME"
os.environ["JIRA_INSTANCE_URL"] = "YOUR_JIRA_INSTANCE_URL"


class PromptFactory():
    jira_template = """
        You are an AI assistant can creat Jira tickets. You can open a ticket related to a specific project,issue type, summary, description, and priority.
        A ticket can be opened within a particular project and issue type. use the following information as default. 
        the project key is "GEN" , issue type is "Requests" and no need for distination.
        Based on this information, I need to open a ticket for 
        {input}."""
    prompt_infos = [
        {
            "name": "jira",
            "description": "open a Jira tickets when requested",
            "prompt_template": jira_template,
        }
    ]


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


class Config():
    model = model_option
    memory = ConversationBufferMemory()
    llm = ChatVertexAI(model_name=model, memory=memory, streaming=True)


cfg = Config()

# Initialize Langchain components
jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
agent = initialize_agent(toolkit.get_tools(
), llm=cfg.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


def generate_destination_chains():
    """
    Creates a list of LLM chains with different prompt templates.
    """
    prompt_factory = PromptFactory()
    destination_chains = {}
    for p_info in prompt_factory.prompt_infos:
        name = p_info['name']
        prompt_template = p_info['prompt_template']
        chain = LLMChain(
            llm=cfg.llm,
            prompt=PromptTemplate(template=prompt_template, input_variables=['input']))
        destination_chains[name] = chain

    default_chain = ConversationChain(
        llm=cfg.llm, verbose=True, output_key="text")
    return prompt_factory.prompt_infos, destination_chains, default_chain


def generate_router_chain(prompt_infos, destination_chains, default_chain):
    """
    Generats the router chains from the prompt infos.
    :param prompt_infos The prompt informations generated above.
    """
    memory = ConversationBufferMemory()
    default_chain = ConversationChain(
        llm=cfg.llm, memory=memory, verbose=True, output_key="text")
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = '\n'.join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=['input'],
        output_parser=RouterOutputParser()
    )
    router_chain = LLMRouterChain.from_llm(cfg.llm, router_prompt)
    return MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )


class ContextChatbot():
    @st.cache_resource
    def setup_chain(_self):
        prompt_infos, destination_chains, default_chain = generate_destination_chains()
        chain = generate_router_chain(
            prompt_infos, destination_chains, default_chain)
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
                print(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})


if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
