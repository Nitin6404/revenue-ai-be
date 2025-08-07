from typing import Dict, Any, List, Optional, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory
import re
import json
from datetime import datetime
import logging
from .query_parser import QueryParser
from .data_fetcher import run_sql, fetch_metric
from .forecast_agent import forecast_revenue
from .insight_agent import get_insight, get_visualization
# from .visualizer import get_visualization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPromptTemplate(StringPromptTemplate):
    """Custom prompt template for the agent"""
    
    template: str
    tools: list
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        # Set the agent_scratchpad variable to be the thoughts
        kwargs["agent_scratchpad"] = thoughts
        
        # Format the tools list
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        
        return self.template.format(**kwargs)

class AgentOrchestrator:
    """Orchestrates the flow between different agents and tools"""
    
    def __init__(self, db_uri: str):
        """Initialize the orchestrator with database connection"""
        self.db = SQLDatabase.from_uri(
            db_uri,
            include_tables=["revenue"],
            custom_table_info={
                "revenue": """
                CREATE TABLE revenue (
                    id SERIAL PRIMARY KEY,
            year INTEGER,
            month VARCHAR,
            department VARCHAR,
            revenue DOUBLE PRECISION
        );

        Example rows:
        (2021, 'January', 'Sales', 270252.05)
        (2021, 'January', 'Marketing', 366988.10)
        """
            }
        )

        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        self.query_parser = QueryParser()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )
        self.setup_tools()
        self.setup_agent()
    
    def setup_tools(self):
        """Set up the tools available to the agent"""
        # Database tools
        self.db_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Custom tools
        self.tools = [
            Tool(
                name="fetch_metric",
                func=self._fetch_metric_tool,
                description="""Useful for fetching metrics from the database. 
                Input should be a JSON string with 'metric', 'dimensions', 'filters', and 'time_frame' keys."""
            ),
            Tool(
                name="forecast_revenue",
                func=self._forecast_revenue_tool,
                description="""Useful for forecasting revenue. 
                Input should be a JSON string with 'periods' (int), 'freq' (str), and 'filters' (dict)."""
            ),
            Tool(
                name="get_insight",
                func=self._get_insight_tool,
                description="""Useful for getting insights from data. 
                Input should be a JSON string with 'data' and 'query'."""
            ),
            Tool(
                name="visualize_data",
                func=self._visualize_data_tool,
                description="""Useful for visualizing data. 
                Input should be a JSON string with 'data' and 'chart_type'."""
            )
        ]
        
        # Add SQL tools
        self.tools.extend(self.db_toolkit.get_tools())
    
    def setup_agent(self):
        """Set up the agent with tools and prompt template"""
        # Define the prompt template
        template = """
        You are a helpful AI assistant for a business intelligence dashboard. 
        Your goal is to help users analyze their data by using the available tools.
        
        You have access to the following tools:
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Previous conversation history:
        {chat_history}
        
        Question: {input}
        {agent_scratchpad}"""
        
        # Create the prompt
        prompt = CustomPromptTemplate(
            template=template,
            tools=self.tools,
            input_variables=["input", "chat_history", "intermediate_steps"]
        )
        
        # Create the output parser
        output_parser = CustomOutputParser()
        
        # Create the LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Define which tools the agent can use
        tool_names = [tool.name for tool in self.tools]
        
        # Create the agent
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=tool_names,
            verbose=True
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            max_iterations=5
        )
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the agent system"""
        try:
            # First, parse the query to understand the intent
            parsed_query = self.query_parser.parse_query(query)
            
            # Add the parsed query to the context
            context = {
                "parsed_query": parsed_query,
                "original_query": query,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Execute the agent with the query
            # result = await self.agent_executor.arun(input=query, context=context)

            # Combine everything into a single input string
            combined_input = f"{query}\nContext: {json.dumps(context)}"
            
            result = await self.agent_executor.arun(input=combined_input)
            
            return {
                "result": result,
                "context": context,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "status": "error"
            }
    
    # Tool methods
    def _fetch_metric_tool(self, input_str: str) -> str:
        """Tool for fetching metrics from the database"""
        try:
            params = json.loads(input_str)
            result = fetch_metric(
                metric=params.get("metric"),
                dimensions=params.get("dimensions", []),
                filters=params.get("filters", {}),
                limit=params.get("limit", 1000),
                return_df=True
            )
            return json.dumps(result.to_dict(orient="records"))
        except Exception as e:
            logger.error(f"Error in fetch_metric_tool: {str(e)}")
            return f"Error: {str(e)}"
    
    def _forecast_revenue_tool(self, input_str: str) -> str:
        """Tool for forecasting revenue"""
        try:
            params = json.loads(input_str)
            result = forecast_revenue(
                periods=params.get("periods", 12),
                freq=params.get("freq", "M"),
                filters=params.get("filters", {})
            )
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error in forecast_revenue_tool: {str(e)}")
            return f"Error: {str(e)}"
    
    def _get_insight_tool(self, input_str: str) -> str:
        """Tool for getting insights from data"""
        try:
            params = json.loads(input_str)
            result = get_insight(
                parsed_query=params.get("parsed_query", {}),
                data=params.get("data", [])
            )
            return result
        except Exception as e:
            logger.error(f"Error in get_insight_tool: {str(e)}")
            return f"Error: {str(e)}"
    
    def _visualize_data_tool(self, input_str: str) -> str:
        """Tool for visualizing data"""
        try:
            params = json.loads(input_str)
            result = get_visualization(
                parsed_query=params.get("parsed_query", {}),
                data=params.get("data", [])
            )
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error in visualize_data_tool: {str(e)}")
            return f"Error: {str(e)}"

class CustomOutputParser(AgentOutputParser):
    """Custom output parser for the agent"""
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output
            )
            
        # Parse the action and action input
        regex = r"Action\s*\d*\s*:([^\n]*)"
        action_match = re.search(regex, llm_output, re.DOTALL)
        
        if action_match is None:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output
            )
            
        action = action_match.group(1).strip()
        action = action.strip('"\'').strip()
        
        # Parse the action input
        input_regex = r"Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        input_match = re.search(input_regex, llm_output, re.DOTALL)
        
        action_input = input_match.group(1).strip() if input_match else ""
        action_input = action_input.strip('"\'').strip()
        
        return AgentAction(
            tool=action, 
            tool_input=action_input,
            log=llm_output
        )
