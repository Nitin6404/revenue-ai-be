"""
Callback handlers for LangChain tools and agents.

This module provides custom callback handlers for monitoring and logging 
the execution of LangChain tools and agents.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

# Configure logging
logger = logging.getLogger(__name__)

class ToolExecutionCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for tool execution events.
    
    This handler logs tool execution details including start/end times,
    inputs, outputs, and any errors that occur.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize the callback handler.
        
        Args:
            session_id: Optional session ID for correlating logs.
        """
        self.session_id = session_id
        self.current_tool = None
        self.tool_start_time = None
        self.tool_input = None
    
    def on_tool_start(
        self, 
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ) -> None:
        """Run when a tool starts running."""
        self.current_tool = serialized.get("name", "unknown_tool")
        self.tool_start_time = datetime.utcnow()
        self.tool_input = input_str
        
        logger.info(
            f"Tool started: {self.current_tool}",
            extra={
                "tool": self.current_tool,
                "input": input_str,
                "session_id": self.session_id,
                "event": "tool_start"
            }
        )
    
    def on_tool_end(
        self, 
        output: str,
        **kwargs
    ) -> None:
        """Run when a tool ends running."""
        if not self.current_tool or not self.tool_start_time:
            return
            
        duration = (datetime.utcnow() - self.tool_start_time).total_seconds()
        
        try:
            # Try to parse the output as JSON for better logging
            output_data = json.loads(output)
            output_str = json.dumps(output_data, indent=2)
            status = output_data.get("status", "unknown")
        except (json.JSONDecodeError, AttributeError):
            output_str = str(output)
            status = "success" if not isinstance(output, Exception) else "error"
        
        logger.info(
            f"Tool completed: {self.current_tool} (status: {status}, duration: {duration:.2f}s)",
            extra={
                "tool": self.current_tool,
                "status": status,
                "duration_seconds": duration,
                "input": self.tool_input,
                "output": output_str[:1000],  # Limit output length
                "session_id": self.session_id,
                "event": "tool_end"
            }
        )
        
        # Reset state
        self.current_tool = None
        self.tool_start_time = None
        self.tool_input = None
    
    def on_tool_error(
        self, 
        error: Union[Exception, KeyboardInterrupt],
        **kwargs
    ) -> None:
        """Run when a tool errors."""
        tool_name = self.current_tool or "unknown_tool"
        duration = (
            (datetime.utcnow() - self.tool_start_time).total_seconds() 
            if self.tool_start_time else 0
        )
        
        logger.error(
            f"Tool error in {tool_name}: {str(error)}",
            extra={
                "tool": tool_name,
                "error": str(error),
                "duration_seconds": duration,
                "input": self.tool_input,
                "session_id": self.session_id,
                "event": "tool_error"
            },
            exc_info=error
        )
        
        # Reset state
        self.current_tool = None
        self.tool_start_time = None
        self.tool_input = None


class AgentCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for agent execution events.
    
    This handler logs agent actions, thoughts, and final outputs.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize the callback handler.
        
        Args:
            session_id: Optional session ID for correlating logs.
        """
        self.session_id = session_id
        self.agent_start_time = None
        self.agent_actions = []
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ) -> None:
        """Run when LLM starts running."""
        if not self.agent_start_time:
            self.agent_start_time = datetime.utcnow()
            
            logger.info(
                "Agent LLM started",
                extra={
                    "session_id": self.session_id,
                    "event": "agent_llm_start",
                    "prompt_count": len(prompts)
                }
            )
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when LLM ends running."""
        if not self.agent_start_time:
            return
            
        duration = (datetime.utcnow() - self.agent_start_time).total_seconds()
        
        logger.info(
            f"Agent LLM completed in {duration:.2f}s",
            extra={
                "session_id": self.session_id,
                "event": "agent_llm_end",
                "duration_seconds": duration,
                "generations": len(response.generations) if response.generations else 0
            }
        )
    
    def on_llm_error(
        self, 
        error: Union[Exception, KeyboardInterrupt],
        **kwargs
    ) -> None:
        """Run when LLM errors."""
        logger.error(
            f"Agent LLM error: {str(error)}",
            extra={
                "session_id": self.session_id,
                "event": "agent_llm_error",
                "error": str(error)
            },
            exc_info=error
        )
    
    def on_agent_action(
        self, 
        action: AgentAction,
        **kwargs
    ) -> Any:
        """Run on agent action."""
        self.agent_actions.append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(
            f"Agent action: {action.tool}",
            extra={
                "session_id": self.session_id,
                "event": "agent_action",
                "tool": action.tool,
                "tool_input": str(action.tool_input)[:500],  # Limit input length
                "log": action.log
            }
        )
    
    def on_agent_finish(
        self, 
        finish: AgentFinish,
        **kwargs
    ) -> None:
        """Run when agent finishes."""
        duration = (
            (datetime.utcnow() - self.agent_start_time).total_seconds()
            if self.agent_start_time else 0
        )
        
        logger.info(
            f"Agent finished in {duration:.2f}s",
            extra={
                "session_id": self.session_id,
                "event": "agent_finish",
                "duration_seconds": duration,
                "output": str(finish.return_values.get("output", ""))[:1000],  # Limit output length
                "action_count": len(self.agent_actions)
            }
        )
        
        # Reset state
        self.agent_start_time = None
        self.agent_actions = []
    
    def on_agent_error(
        self, 
        error: Union[Exception, KeyboardInterrupt],
        **kwargs
    ) -> None:
        """Run when agent errors."""
        duration = (
            (datetime.utcnow() - self.agent_start_time).total_seconds()
            if self.agent_start_time else 0
        )
        
        logger.error(
            f"Agent error after {duration:.2f}s: {str(error)}",
            extra={
                "session_id": self.session_id,
                "event": "agent_error",
                "duration_seconds": duration,
                "error": str(error),
                "action_count": len(self.agent_actions)
            },
            exc_info=error
        )
        
        # Reset state
        self.agent_start_time = None
        self.agent_actions = []


def get_default_callbacks(session_id: Optional[str] = None) -> List[BaseCallbackHandler]:
    """Get the default callback handlers for tool and agent execution.
    
    Args:
        session_id: Optional session ID for correlating logs.
        
    Returns:
        A list of callback handlers.
    """
    return [
        ToolExecutionCallbackHandler(session_id=session_id),
        AgentCallbackHandler(session_id=session_id)
    ]
