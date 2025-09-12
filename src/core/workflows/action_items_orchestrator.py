"""Action Items Orchestrator Workflow.

This workflow orchestrates the complete action items processing pipeline by composing
sub-workflows. It provides the main entry point and maintains backward compatibility
with the original ActionItemsWorkflow API.
"""

from typing import Any

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from src.core.workflows.models import ActionItemsList, AgentExecutionResult
from src.core.workflows.sub_workflows.action_items_generation_workflow import (
    ActionItemsGenerationWorkflow,
)
from src.core.workflows.sub_workflows.agent_dispatch_workflow import (
    AgentDispatchWorkflow,
)
from src.core.workflows.sub_workflows.meeting_notes_workflow import MeetingNotesWorkflow
from src.infrastructure.logging.logging_config import get_logger

logger = get_logger("workflows.action_items_orchestrator")


class MeetingNotesRetrieved(Event):
    """Event indicating meeting notes have been retrieved."""

    meeting_notes: str


class ActionItemsProcessed(Event):
    """Event indicating action items have been generated and validated."""

    action_items: ActionItemsList


class ActionItemsOrchestrator(Workflow):
    """Orchestrator workflow that composes sub-workflows for action items processing.

    This workflow provides:
    1. Backward compatibility with the original ActionItemsWorkflow API
    2. Composition of focused sub-workflows
    3. Centralized error handling and logging
    """

    def __init__(
        self,
        llm,
        max_iterations: int = 3,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the orchestrator workflow.

        Args:
            llm: Language model for sub-workflows
            max_iterations: Maximum number of refinement iterations
        """
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.max_iterations = max_iterations

        logger.info(
            f"Initialized ActionItemsOrchestrator with max_iterations: {max_iterations}"
        )

    @step
    async def retrieve_meeting_notes(
        self, event: StartEvent
    ) -> MeetingNotesRetrieved | StopEvent:
        """Retrieve meeting notes using the MeetingNotesWorkflow.

        This step maintains compatibility with the original workflow API.
        """
        logger.info(
            f"Retrieving meeting notes for meeting: {event['meeting']}, "
            f"date: {event['date']}"
        )

        try:
            # Initialize and run the meeting notes workflow
            meeting_notes_workflow = MeetingNotesWorkflow(llm=self.llm)

            meeting_notes_result = await meeting_notes_workflow.run(
                date=event.date, meeting=event.meeting
            )

            # Handle workflow result
            if hasattr(meeting_notes_result, "error") and meeting_notes_result.error:
                logger.error(
                    f"Meeting notes workflow failed: {meeting_notes_result.result}"
                )
                return StopEvent(result=meeting_notes_result.result, error=True)

            # Extract content
            meeting_notes_content = (
                meeting_notes_result.result
                if hasattr(meeting_notes_result, "result")
                else str(meeting_notes_result)
            )

            # Validate content
            if (
                not isinstance(meeting_notes_content, str)
                or not meeting_notes_content.strip()
            ):
                logger.warning("Meeting notes content is empty or invalid")
                return StopEvent(result="empty_meeting_notes", error=True)

            logger.info(
                "Successfully retrieved meeting notes "
                f"({len(meeting_notes_content)} chars)"
            )

            return MeetingNotesRetrieved(meeting_notes=meeting_notes_content)

        except Exception as e:
            logger.error(f"Error retrieving meeting notes: {e}")
            return StopEvent(result="meeting_notes_error", error=True)

    @step
    async def generate_action_items(
        self, event: MeetingNotesRetrieved
    ) -> ActionItemsProcessed | StopEvent:
        """Generate action items using the ActionItemsGenerationWorkflow."""

        logger.info("Generating action items from meeting notes")

        try:
            # Initialize and run the action items generation workflow
            generation_workflow = ActionItemsGenerationWorkflow(
                llm=self.llm, max_iterations=self.max_iterations
            )

            generation_result = await generation_workflow.run(
                meeting_notes=event.meeting_notes
            )

            # Extract action items from result (now returns StopEvent directly)
            if hasattr(generation_result, "result"):
                action_items = generation_result.result
            else:
                logger.error("Invalid result from action items generation workflow")
                return StopEvent(result="generation_error", error=True)

            logger.info(f"Generated {len(action_items.action_items)} action items")

            return ActionItemsProcessed(action_items=action_items)

        except Exception as e:
            logger.error(f"Error generating action items: {e}")
            return StopEvent(result="generation_error", error=True)

    @step
    async def dispatch_to_agents(self, event: ActionItemsProcessed) -> StopEvent:
        """Dispatch action items to agents using the AgentDispatchWorkflow."""

        logger.info("Dispatching action items to agents")

        try:
            # Initialize and run the agent dispatch workflow
            dispatch_workflow = AgentDispatchWorkflow(llm=self.llm)

            dispatch_result = await dispatch_workflow.run(
                action_items=event.action_items
            )

            # Handle results
            if hasattr(dispatch_result, "error") and dispatch_result.error:
                logger.error(f"Agent dispatch failed: {dispatch_result.result}")
                return StopEvent(result=dispatch_result.result, error=True)

            # Extract execution results
            execution_results = (
                dispatch_result.result
                if hasattr(dispatch_result, "result")
                else dispatch_result
            )

            # Compile final summary
            if isinstance(execution_results, list):
                successful_count = sum(
                    1
                    for result in execution_results
                    if isinstance(result, AgentExecutionResult) and result.success
                )

                logger.info(
                    f"Action items processing completed: "
                    f"{successful_count}/{len(execution_results)} successful executions"
                )

            return StopEvent(result=execution_results)

        except Exception as e:
            logger.error(f"Error dispatching to agents: {e}")
            return StopEvent(result="dispatch_error", error=True)
