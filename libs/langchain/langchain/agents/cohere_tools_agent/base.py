from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel
from typing import Any, List, Optional, Sequence, Tuple, Type, Union, Dict, Callable
import json
from json import JSONDecodeError
from typing import List, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, Generation

from langchain.agents.agent import MultiActionAgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

def create_cohere_tools_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    """Create an agent that uses OpenAI tools.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:

        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_openai_tools_agent

            prompt = hub.pull("hwchase17/openai-tools-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_openai_tools_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            ``MessagesPlaceholder``. Intermediate agent actions and tool output
            messages will be passed in here.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
    """
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(tools=[convert_to_cohere_tool(tool) for tool in tools])

    agent = (
        RunnablePassthrough.assign(
            tool_results=lambda x: format_to_cohere_tools_messages(
                x["intermediate_steps"]
            ),
            tools_input_only=lambda x: len(x["intermediate_steps"]) == 0,
            agent_scratchpad = ""
        )
        | prompt
        | llm_with_tools
        | CohereToolsAgentOutputParser()
    )
    return agent

def format_to_cohere_tools_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into Messages.
    """

    messages = []
    for agent_action, observation in intermediate_steps:
        messages.append(AIMessage(additional_kwargs=
                                  {
                                        "input": {
                                            "tool_name": agent_action.tool,
                                            "tool_input": agent_action.tool_input
                                        },
                                        "result": observation
                                  }))
    return messages

def convert_to_cohere_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an Cohere tool.
    """
    if isinstance(tool, BaseTool):
        return {
            "name": tool.name,
            "definition": {
                "description": tool.description,
                "inputs": [{
                    "name": a["title"],
                    "description": a["description"],
                    "type": a["type"],
                } for a in tool.args],
            }
        }
    # todo
    return {"type": "function", "function": function}

class CohereToolsAgentOutputParser(AgentOutputParser):
    """Parses a message into agent actions/finish."""

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if result[0].generation_info["tool_inputs"]: 
            AgentAction(
                tool=result[0].generation_info["tool_name"],
                tool_input=result[0].generation_info["parameters"],
            )
        else:
            AgentFinish(return_values={"output": result[0].text})

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")
