"""Agent Dispatch Workflow.

This workflow handles routing action items to appropriate agents and collecting results.
"""

from typing import List

import requests
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from src.core.workflows.models import (
    ActionItemsList,
    AgentExecutionResult,
    AgentRoutingDecision,
)
from src.infrastructure.logging.logging_config import get_logger
from src.infrastructure.prompts.prompts import (
    AGENT_QUERY_PROMPT,
    TOOL_DISPATCHER_PROMPT,
)
from src.infrastructure.registry.registry_client import get_registry_client

logger = get_logger("workflows.agent_dispatch")


class ActionItemsInput(Event):
    """Input event containing action items to dispatch."""

    action_items: ActionItemsList


class RoutingDecisions(Event):
    """Event containing routing decisions for action items."""

    decisions: List[AgentRoutingDecision]
    action_items: ActionItemsList


class ExecutionRequired(Event):
    """Event indicating agent execution is needed."""

    decision: AgentRoutingDecision
    action_item_data: dict


class ExecutionCompleted(Event):
    """Event indicating agent execution has completed."""

    result: AgentExecutionResult


class AgentDispatchWorkflow(Workflow):
    """Workflow for routing action items to appropriate agents and executing them.

    This workflow uses the agent registry for dynamic agent discovery and
    LLMTextCompletionProgram for structured routing decisions.
    """

    def __init__(self, llm, *args, **kwargs):
        """Initialize the workflow.

        Args:
            llm: Language model for routing decisions
        """
        super().__init__(*args, **kwargs)
        self.llm = llm
        logger.info("Initialized AgentDispatchWorkflow")

    @step
    async def initialize_dispatch(self, event: StartEvent) -> ActionItemsInput:
        """Initialize dispatch workflow with action items from StartEvent."""
        logger.info("Initializing agent dispatch workflow")

        # Extract action items from StartEvent
        action_items = event.action_items
        logger.info(
            f"Received {len(action_items.action_items)} action items for dispatch"
        )
        return ActionItemsInput(action_items=action_items)

    @step
    async def route_action_items(
        self, ctx: Context, event: ActionItemsInput
    ) -> RoutingDecisions | StopEvent:
        """Route action items to appropriate agents using dynamic discovery.

        Uses agent registry to discover available agents and LLM to
        make routing decisions.
        """
        logger.info(f"Routing {len(event.action_items.action_items)} action items")

        # Discover available agents
        try:
            registry_client = get_registry_client()
            available_agents = await registry_client.discover_agents()

            if not available_agents:
                logger.warning("No agents found in registry")
                return StopEvent(result="no_agents_available", error=True)

            logger.info(f"Found {len(available_agents)} available agents")

            # Build agent descriptions for LLM decision making
            agent_descriptions = []
            for agent in available_agents:
                agent_descriptions.append(f"{agent.name}: {agent.description}")

        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
            return StopEvent(result="agent_discovery_error", error=True)

        # Make routing decisions for each action item
        decisions = []

        try:
            # Create structured program for routing decisions
            routing_program = LLMTextCompletionProgram.from_defaults(
                llm=self.llm,
                output_cls=AgentRoutingDecision,
                prompt=TOOL_DISPATCHER_PROMPT,
                verbose=True,
            )

            for idx, action_item in enumerate(event.action_items.action_items):
                try:
                    logger.debug(f"Routing action item {idx}: {action_item.title}")

                    # Get structured routing decision
                    decision = await routing_program.acall(
                        action_item=action_item.model_dump(),
                        agents_list="\\n".join(agent_descriptions),
                        action_item_index=idx,
                    )

                    # Find the selected agent by name
                    selected_agent = self._find_agent_by_name(
                        available_agents, decision.agent_name
                    )

                    if selected_agent:
                        decision.agent_name = selected_agent.agent_id
                        logger.info(
                            f"Routed '{action_item.title}' to {selected_agent.name}"
                        )
                    else:
                        logger.warning(
                            f"Agent '{decision.agent_name}' not "
                            "found, marking as unassigned"
                        )
                        decision.agent_name = "UNASSIGNED_AGENT"

                    decisions.append(decision)

                except Exception as e:
                    logger.error(f"Error routing action item {idx}: {e}")
                    # Create fallback decision
                    decisions.append(
                        AgentRoutingDecision(
                            action_item_index=idx,
                            agent_name="UNASSIGNED_AGENT",
                            routing_reason=f"Error during routing: {str(e)}",
                            requires_human_approval=True,
                        )
                    )

            await ctx.store.set("total_executions", len(decisions))
            logger.info(f"Created {len(decisions)} routing decisions")

            return RoutingDecisions(
                decisions=decisions, action_items=event.action_items
            )

        except Exception as e:
            logger.error(f"Error during routing process: {e}")
            return StopEvent(result="routing_error", error=True)

    @step
    async def execute_action_items(
        self, ctx: Context, event: RoutingDecisions
    ) -> ExecutionRequired | None:
        """Dispatch action items to agents for execution."""

        for decision in event.decisions:
            action_item = event.action_items.action_items[decision.action_item_index]

            ctx.send_event(
                ExecutionRequired(
                    decision=decision, action_item_data=action_item.model_dump()
                )
            )

        return None

    @step
    async def execute_single_action(
        self, event: ExecutionRequired
    ) -> ExecutionCompleted:
        """Execute a single action item via the assigned agent."""

        action_item = event.action_item_data
        decision = event.decision

        logger.info(f"Executing action item via {decision.agent_name}")

        if decision.agent_name == "UNASSIGNED_AGENT":
            return ExecutionCompleted(
                result=AgentExecutionResult(
                    action_item_index=decision.action_item_index,
                    agent_name=decision.agent_name,
                    success=False,
                    result="No suitable agent found for this action item",
                    error_message="Agent routing failed",
                )
            )

        try:
            # Get agent endpoint from registry
            registry_client = get_registry_client()
            available_agents = await registry_client.discover_agents()

            agent_endpoint = None
            for agent in available_agents:
                if agent.agent_id == decision.agent_name:
                    agent_endpoint = agent.endpoint
                    break

            if not agent_endpoint:
                logger.error(f"No endpoint found for agent: {decision.agent_name}")
                return ExecutionCompleted(
                    result=AgentExecutionResult(
                        action_item_index=decision.action_item_index,
                        agent_name=decision.agent_name,
                        success=False,
                        result="Agent endpoint not found",
                        error_message=f"No endpoint for agent {decision.agent_name}",
                    )
                )

            # Execute via agent API
            agent_url = f"{agent_endpoint}/agent"

            response = requests.post(
                agent_url,
                json={"query": AGENT_QUERY_PROMPT.format(action_item)},
                timeout=30,
            )
            response.raise_for_status()

            response_data = response.json()
            logger.info(f"Agent {decision.agent_name} completed execution")

            return ExecutionCompleted(
                result=AgentExecutionResult(
                    action_item_index=decision.action_item_index,
                    agent_name=decision.agent_name,
                    success=True,
                    result=str(response_data),
                    error_message=None,
                )
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling agent {decision.agent_name}: {e}")
            return ExecutionCompleted(
                result=AgentExecutionResult(
                    action_item_index=decision.action_item_index,
                    agent_name=decision.agent_name,
                    success=False,
                    result="Agent execution failed",
                    error_message=str(e),
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error executing action item: {e}")
            return ExecutionCompleted(
                result=AgentExecutionResult(
                    action_item_index=decision.action_item_index,
                    agent_name=decision.agent_name,
                    success=False,
                    result="Unexpected execution error",
                    error_message=str(e),
                )
            )

    @step
    async def collect_execution_results(
        self, ctx: Context, event: ExecutionCompleted
    ) -> StopEvent | None:
        """Collect results from all agent executions."""

        total_executions = await ctx.store.get("total_executions")
        results = ctx.collect_events(event, [ExecutionCompleted] * total_executions)

        if results is None:
            return None

        logger.info(f"Collected {len(results)} execution results")

        # Extract results and compile summary
        execution_results = [result.result for result in results]
        successful_executions = sum(1 for result in execution_results if result.success)

        logger.info(
            f"Execution summary: {successful_executions}/"
            f"{len(execution_results)} successful"
        )

        return StopEvent(result=execution_results)

    def _find_agent_by_name(self, agents, agent_name: str):
        """Find agent by name from the available agents list."""
        agent_name = agent_name.lower().strip()

        # Try exact match by name first
        for agent in agents:
            if agent.name.lower() == agent_name:
                return agent

        # Try partial match by name
        for agent in agents:
            if agent_name in agent.name.lower() or agent.name.lower() in agent_name:
                return agent

        return None
