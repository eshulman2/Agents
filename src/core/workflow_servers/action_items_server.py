"""Action items server"""

from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import HTTPException
from langfuse import get_client as get_langfuse_client
from pydantic import BaseModel, PastDate

from src.core.base.base_server import BaseServer
from src.core.workflows.action_items_orchestrator import ActionItemsOrchestrator
from src.infrastructure.config import get_config, get_model
from src.infrastructure.logging.logging_config import get_logger
from src.infrastructure.observability.observability import set_up_langfuse

set_up_langfuse()
logger = get_logger("workflow_server.action_items")
config = get_config()


class Meeting(BaseModel):
    """Request model for meeting information.

    Attributes:
        meeting: Name or identifier of the meeting
        date: Date of the meeting in string format
    """

    meeting: str
    date: PastDate


class ActionItemsResponse(BaseModel):
    """Response model for workflow output.

    Attributes:
        action_items: Dictionary containing structured action items
    """

    action_items: Any


class ActionItemsServer(BaseServer):
    """Action item agent server implementation."""

    def create_service(self, llm):
        logger.info("Creating action items workflow")

        workflow = ActionItemsOrchestrator(
            llm=llm,
            timeout=30,
            verbose=True,
            max_iterations=20,
        )
        logger.info("Action items workflow created successfully")

        return workflow

    def additional_routes(self):

        @self.app.post("/action-items", response_model=ActionItemsResponse)
        async def create_action_items_endpoint(request: Meeting):
            """FastAPI endpoint for processing meeting notes into action items.

            This endpoint initializes and runs the ActionItemsOrchestrator to process
            meeting information and generate structured action items with full
            observability tracking via Langfuse.

            Args:
                request: Meeting request containing meeting name and date

            Returns:
                ActionItemsResponse with generated action items

            Raises:
                HTTPException: If workflow execution fails
            """
            logger.info(
                f"Processing action items request with {len(request.meeting)} "
                "characters of meeting notes"
            )
            try:

                session_id = f"action-items-workflow-{str(uuid4())}"

                langfuse_client = get_langfuse_client()

                with langfuse_client.start_as_current_span(name=session_id) as span:

                    res = await self.service.run(
                        meeting=request.meeting, date=request.date
                    )

                    span.update_trace(
                        session_id=session_id,
                        input=(
                            f"meeting: {request.meeting}, "
                            f"date: {request.date.strftime("%Y-%m-%d")}"
                        ),
                        output=str(res),
                    )
                langfuse_client.flush()

                logger.info("Action items workflow completed successfully")
                return ActionItemsResponse(action_items=res)
            except Exception as e:
                logger.error(f"Error processing action items request: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error processing request: {e}"
                ) from e


# Initialize the server
logger.info("Initializing Action Items Workflow server")
server = ActionItemsServer(
    llm=get_model(config.config),
    title="Action Items Workflow",
    description=("llamaindex workflow for taking meeting notes and process them"),
)
app = server.app
logger.info("Action Items Workflow server initialized successfully")

if __name__ == "__main__":
    uvicorn.run(app, host=config.config.host, port=config.config.port, log_level="info")
