"""
Agent definitions and setup for the RAG system.
This module provides functionality for creating and configuring the agents used in the RAG pipeline.
"""

from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_gigachat import GigaChat
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.prebuilt import create_react_agent

from app.config.settings import (
    DEFAULT_LLM_MODEL,
    LLM_SCOPE,
    LLM_TIMEOUT,
    SUPERVISOR_PROMPT,
    SEARCHER_PROMPT,
)


class AgentManager:
    """
    Manages the creation and configuration of agents for the RAG system.
    
    This class handles the setup of the research and answer agents,
    as well as the creation of the workflow graph.
    """

    def __init__(
        self,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        credentials: str = "",
    ) -> None:
        """
        Initialize the agent manager.

        Args:
            llm_model_name: Name of the language model to use.
            credentials: Credentials for the language model.
        """
        self.llm = GigaChat(
            scope=LLM_SCOPE,
            verify_ssl_certs=False,
            model=llm_model_name,
            credentials=credentials,
            profanity_check=False,
            timeout=LLM_TIMEOUT,
        )

    def create_rag_tool(self, vector_store: Any) -> tool:
        """
        Create the RAG tool for the research agent.

        Args:
            vector_store: The vector store instance to use for searching.

        Returns:
            A tool that can be used by the research agent.
        """
        @tool
        def rag_tool(query_text: str) -> str:
            """Tool to retrieve relevant information from the vector store"""
            dense_vector, sparse_vector = vector_store.embedding_manager.get_embeddings(query_text)
            results = vector_store.hybrid_search(dense_vector, sparse_vector)
            return results[0]['text'] if results else "No relevant information found."

        return rag_tool

    def setup_workflow(self, vector_store: Any) -> StateGraph:
        """
        Set up the complete workflow graph with research and answer agents.

        Args:
            vector_store: The vector store instance to use for searching.

        Returns:
            A compiled workflow graph.
        """
        # Create the research agent
        research_agent = create_react_agent(
            model=self.llm,
            tools=[self.create_rag_tool(vector_store)],
            prompt=SEARCHER_PROMPT,
        )

        # Create the answer agent
        answer_agent = create_react_agent(
            model=self.llm,
            tools=[],
            prompt=SUPERVISOR_PROMPT,
        )

        # Create and configure the workflow graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("research", research_agent)
        workflow.add_node("answer", answer_agent)
        workflow.add_edge(START, "research")
        workflow.add_edge("research", "answer")
        workflow.add_edge("answer", END)

        return workflow.compile() 