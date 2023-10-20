import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.agents import AgentToolkit
from langchain.llms import OpenAI
from langchain.utilities.dockerhub import DockerhubAPIWrapper
from langchain.agents.agent_toolkits.dockerhub.toolkit import DockerhubToolkit

# Initialize Langchain components
llm = OpenAI(temperature=0)
dockerhub = DockerhubAPIWrapper()
toolkit = DockerhubToolkit.from_dockerhub_api_wrapper(dockerhub)
agent = initialize_agent(toolkit.get_tools(
), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Streamlit app
st.title("Dockerhub Image Search")

# User input
search_query = st.text_input("Enter a search query")

# Search for Docker images in Dockerhub
if search_query:
    response = agent.run(f"search Docker images for {search_query}")

    # Process the response to extract the relevant information
    images = response.split(
        "Here are some Docker images that match your search query:")[1].split("\n")
    images = [image.strip() for image in images if image.strip()]

    # Display the Docker images
    st.subheader("Docker images in Dockerhub:")
    for image in images:
        st.write(image)
