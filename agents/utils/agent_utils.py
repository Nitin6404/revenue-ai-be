"""
Utility functions for working with LangChain agents.

This module provides helper functions for initializing, configuring, 
and executing LangChain agents with consistent behavior.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langchain.agents import AgentExecutor, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseLanguageModel, BaseMessage
from langchain.callbacks.base import BaseCallbackManager

from .callback_handlers import get_default_callbacks
from ..tools import get_all_tools, get_tool_by_name

# Configure logging
logger = logging.getLogger(__name__)

def initialize_llm(
    model_name: str = "gpt-3.5-turbo-16k",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    **kwargs
) -> ChatOpenAI:
    """Initialize a language model for the agent.
    
    Args:
        model_name: The name of the model to use.
        temperature: The temperature to use for generation.
        max_tokens: Maximum number of tokens to generate.
        **kwargs: Additional keyword arguments to pass to the model.
        
    Returns:
        An initialized language model.
    """
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=60,  # seconds
        **kwargs
    )

def create_conversation_memory(
    memory_key: str = "chat_history",
    return_messages: bool = True,
    **kwargs
) -> ConversationBufferMemory:
    """Create a conversation memory for the agent.
    
    Args:
        memory_key: The key to use for storing the chat history.
        return_messages: Whether to return messages or strings.
        **kwargs: Additional keyword arguments for the memory.
        
    Returns:
        A conversation memory instance.
    """
    return ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=return_messages,
        **kwargs
    )

def initialize_agent_executor(
    tools: List[Tool],
    llm: BaseLanguageModel,
    agent_type: Union[AgentType, str] = AgentType.OPENAI_FUNCTIONS,
    memory: Optional[Any] = None,
    callbacks: Optional[List[Any]] = None,
    verbose: bool = False,
    max_iterations: int = 10,
    early_stopping_method: str = "generate",
    **kwargs
) -> AgentExecutor:
    """Initialize an agent executor with the specified tools and configuration.
    
    Args:
        tools: List of tools the agent can use.
        llm: The language model to use.
        agent_type: The type of agent to create.
        memory: Optional memory for the agent.
        callbacks: Optional callbacks for the agent.
        verbose: Whether to enable verbose output.
        max_iterations: Maximum number of iterations the agent can take.
        early_stopping_method: Method for early stopping.
        **kwargs: Additional keyword arguments for the agent.
        
    Returns:
        An initialized agent executor.
    """
    from langchain.agents import initialize_agent
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=agent_type,
        memory=memory,
        callbacks=callbacks,
        verbose=verbose,
        **kwargs
    )
    
    # Set max iterations
    if hasattr(agent, 'max_iterations'):
        agent.max_iterations = max_iterations
    
    # Set early stopping method
    if hasattr(agent, 'early_stopping_method'):
        agent.early_stopping_method = early_stopping_method
    
    return agent

def execute_agent(
    agent: AgentExecutor,
    input_text: str,
    session_id: Optional[str] = None,
    callbacks: Optional[List[Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Execute an agent with the given input and return the result.
    
    Args:
        agent: The agent executor to run.
        input_text: The input text to process.
        session_id: Optional session ID for logging.
        callbacks: Optional callbacks for the execution.
        **kwargs: Additional keyword arguments for the agent's run method.
        
    Returns:
        A dictionary containing the agent's response and metadata.
    """
    from langchain.schema import AgentFinish
    
    # Set up default callbacks if none provided
    if callbacks is None:
        callbacks = get_default_callbacks(session_id=session_id)
    
    try:
        # Execute the agent
        result = agent.run(input_text, callbacks=callbacks, **kwargs)
        
        # Format the response
        if isinstance(result, AgentFinish):
            output = result.return_values.get("output", "")
            log = result.log
        else:
            output = str(result)
            log = output
        
        return {
            "status": "success",
            "output": output,
            "log": log,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(
            f"Error executing agent: {str(e)}",
            extra={"session_id": session_id},
            exc_info=True
        )
        
        return {
            "status": "error",
            "error": str(e),
            "session_id": session_id
        }

def format_agent_response(
    response: Dict[str, Any],
    include_metadata: bool = False
) -> Dict[str, Any]:
    """Format an agent's response for API output.
    
    Args:
        response: The agent's response dictionary.
        include_metadata: Whether to include metadata in the response.
        
    Returns:
        A formatted response dictionary.
    """
    if response.get("status") == "error":
        return {
            "status": "error",
            "message": response.get("error", "An unknown error occurred"),
            "session_id": response.get("session_id")
        }
    
    formatted = {
        "status": "success",
        "result": response.get("output", ""),
        "session_id": response.get("session_id")
    }
    
    if include_metadata:
        formatted["metadata"] = {
            "log": response.get("log", ""),
            "timestamp": response.get("timestamp")
        }
    
    return formatted

def get_available_tools() -> List[Dict[str, Any]]:
    """Get a list of all available tools with their descriptions.
    
    Returns:
        A list of dictionaries containing tool information.
    """
    tools = []
    for tool in get_all_tools():
        tool_info = {
            "name": tool.name,
            "description": tool.description,
            "args": {}
        }
        
        # Add argument schema if available
        if hasattr(tool, 'args_schema'):
            schema = tool.args_schema.schema()
            if "properties" in schema:
                tool_info["args"] = schema["properties"]
        
        tools.append(tool_info)
    
    return tools
