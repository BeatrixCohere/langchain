from langchain.retrievers import CohereRagRetriever

from langchain_community.chat_models import ChatCohere
from langchain.agents.cohere_tools_agent import create_cohere_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import WikipediaRetriever
from langchain import hub
from langchain.agents import AgentExecutor


# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

chat = ChatCohere(cohere_api_key="API KEY")

retriever = WikipediaRetriever()
retriever_tool = create_retriever_tool(
    retriever,
    "wikipedia",
    "Search for information on Wikipedia",
)
agent = create_cohere_tools_agent(
    model=chat,
    tools=[retriever_tool],
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

agent_executor.invoke({"input": "What is the highest mountain in the world?"})


