"""Action Items Generation Workflow.

This workflow handles the generation and refinement of action items from meeting notes
using LLMTextCompletionProgram with Pydantic models for structured output.
"""

from llama_index.core.memory import Memory
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from src.core.workflows.models import ActionItemsList, ReviewFeedback
from src.infrastructure.logging.logging_config import get_logger
from src.infrastructure.prompts.prompts import (
    ACTION_ITEMS_PROMPT,
    REFINEMENT_PROMPT,
    REVIEWER_PROMPT,
)

logger = get_logger("workflows.action_items_generation")


class ReviewRequired(Event):
    """Event indicating review is needed."""

    action_items: ActionItemsList
    meeting_notes: str


class RefinementRequired(Event):
    """Event indicating refinement based on review feedback."""

    action_items: ActionItemsList
    feedback: ReviewFeedback
    meeting_notes: str


class ActionItemsGenerationWorkflow(Workflow):
    """Workflow for generating and refining action items from meeting notes.

    This workflow uses LLMTextCompletionProgram with Pydantic models to ensure
    structured, validated output without manual JSON parsing.
    """

    def __init__(self, llm, *args, max_iterations: int = 20, **kwargs):
        """Initialize the workflow.

        Args:
            llm: Language model for generating action items
            max_iterations: Maximum number of refinement iterations
        """
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.max_iterations = max_iterations
        self.memory = Memory.from_defaults(
            session_id="action_items_generation", token_limit=40000
        )
        logger.info(
            f"Initialized ActionItemsGenerationWorkflow with "
            f"max_iterations: {max_iterations}"
        )

    @step
    async def generate_action_items(
        self, ctx: Context, event: StartEvent
    ) -> ReviewRequired:
        """Generate initial action items from meeting notes.

        Uses LLMTextCompletionProgram to ensure structured output via Pydantic models.
        """
        logger.info("Generating action items from meeting notes")

        try:
            # Create structured program for action items generation
            program = LLMTextCompletionProgram.from_defaults(
                llm=self.llm,
                output_cls=ActionItemsList,
                prompt=ACTION_ITEMS_PROMPT,
                verbose=True,
            )

            # Generate action items with structured output
            action_items = await program.acall(meeting_notes=event.meeting_notes)

            # Validate the generated action items
            if not isinstance(action_items, ActionItemsList):
                logger.error("Generated output is not ActionItemsList type")
                raise ValueError("Invalid action items structure generated")

            if not action_items.action_items:
                logger.warning("No action items were generated")
                # Return empty but valid ActionItemsList
                action_items = ActionItemsList(action_items=[])

            logger.info(f"Generated {len(action_items.action_items)} action items")
            logger.debug(
                f"Action items: {[item.title for item in action_items.action_items]}"
            )

            # Store iteration count for refinement tracking
            await ctx.store.set("iteration_count", 0)

            return ReviewRequired(
                action_items=action_items, meeting_notes=event.meeting_notes
            )

        except Exception as e:
            logger.error(f"Error generating action items: {e}")
            raise

    @step
    async def review_action_items(
        self, event: ReviewRequired
    ) -> StopEvent | RefinementRequired:
        """Review generated action items for quality and completeness.

        Uses LLMTextCompletionProgram for structured review feedback.
        """
        logger.info("Reviewing generated action items")

        try:
            # Create structured program for review with improved prompt handling
            review_program = LLMTextCompletionProgram.from_defaults(
                llm=self.llm,
                output_cls=ReviewFeedback,
                prompt=REVIEWER_PROMPT,
                verbose=True,
            )

            # Get structured review feedback
            review = await review_program.acall(
                action_items=event.action_items.model_dump_json(indent=2),
                meeting_notes=event.meeting_notes,
            )

            # Validate review result
            if not isinstance(review, ReviewFeedback):
                logger.warning(
                    "Review output is not ReviewFeedback type, proceeding with original"
                )
                return StopEvent(result=event.action_items)

            if not review.requires_changes:
                logger.info("Review passed: Action items approved")
                print(event.action_items)
                return StopEvent(result=event.action_items)

            logger.info(f"Review identified issues: {review.feedback}")
            return RefinementRequired(
                action_items=event.action_items,
                feedback=review,
                meeting_notes=event.meeting_notes,
            )

        except Exception as e:
            logger.error(f"Error during review: {e}")
            # If review fails, proceed with current action items
            logger.warning("Review failed, proceeding with current action items")
            return StopEvent(result=event.action_items)

    @step
    async def refine_action_items(
        self, ctx: Context, event: RefinementRequired
    ) -> ReviewRequired | StopEvent:
        """Refine action items based on review feedback.

        Uses memory to maintain context across refinement iterations.
        """
        iteration_count = await ctx.store.get("iteration_count", default=0)
        iteration_count += 1

        logger.info(
            f"Refining action items (iteration "
            f"{iteration_count}/{self.max_iterations})"
        )

        if iteration_count >= self.max_iterations:
            logger.warning(
                "Max refinement iterations reached, proceeding with "
                "current action items"
            )
            return StopEvent(result=event.action_items)

        await ctx.store.set("iteration_count", iteration_count)

        try:
            # Create structured program for refinement
            refinement_program = LLMTextCompletionProgram.from_defaults(
                llm=self.llm,
                output_cls=ActionItemsList,
                prompt=REFINEMENT_PROMPT,
                verbose=True,
            )

            # Refine with structured output
            refined_action_items = await refinement_program.acall(
                review=event.feedback.feedback,
                action_items=event.action_items.model_dump_json(indent=2),
            )

            # Validate the refined action items
            if not isinstance(refined_action_items, ActionItemsList):
                logger.warning("Refined output is not ActionItemsList, using original")
                return StopEvent(result=event.action_items)

            if not refined_action_items.action_items:
                logger.warning("Refined output has no action items, using original")
                return StopEvent(result=event.action_items)

            logger.info(f"Refined action items (iteration {iteration_count})")
            logger.debug(
                f"Refined items: "
                f"{[item.title for item in refined_action_items.action_items]}"
            )

            return ReviewRequired(
                action_items=refined_action_items, meeting_notes=event.meeting_notes
            )

        except Exception as e:
            logger.error(f"Error during refinement: {e}")
            # If refinement fails, return original action items
            logger.warning("Refinement failed, returning original action items")
            return StopEvent(result=event.action_items)
