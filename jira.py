import os
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.llms import OpenAI
from langchain.utilities.jira import JiraAPIWrapper

# Set up Jira authentication
os.environ["JIRA_API_TOKEN"] = "YOUR_JIRA_API_TOKEN"
os.environ["JIRA_USERNAME"] = "YOUR_JIRA_USERNAME"
os.environ["JIRA_INSTANCE_URL"] = "YOUR_JIRA_INSTANCE_URL"

# Initialize Langchain components
llm = OpenAI(temperature=0)
jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
agent = initialize_agent(toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Create a new issue in Jira
issue_summary = "New issue summary"
issue_description = "New issue description"
project_key = "PROJECT_KEY"  # Replace with the key of your Jira project
issue_type = "Task"
priority = "Low"

response = agent.run(f"make a new issue in project {project_key} with summary {issue_summary} and description {issue_description}")

# Process the response to extract the relevant information
issue_key = response.split('with the summary "')[1].split('" and description')[0]

print(f"A new issue has been created in Jira with key: {issue_key}")