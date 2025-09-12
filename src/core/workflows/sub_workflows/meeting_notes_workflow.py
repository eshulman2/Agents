"""Meeting Notes Workflow for extracting meeting notes.

This module implements a workflow that:
1. Retrieves calendar events for a specific date and meeting name
2. Extracts Google Doc attachments from the meeting
3. Identifies which attachment contains meeting notes
4. Fetches and returns the content of the meeting notes document
"""

from uuid import uuid4

import nest_asyncio
from langfuse import get_client as get_langfuse_client
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.tools.mcp import BasicMCPClient
from pydantic import BaseModel

from src.infrastructure.config import get_config
from src.infrastructure.logging.logging_config import get_logger
from src.infrastructure.prompts.prompts import IDENTIFY_MEETING_NOTES

logger = get_logger("workflows.meeting_notes_workflow")
config = get_config()

nest_asyncio.apply()


class FileToId(BaseModel):
    """Model for representing a file with its title and ID.

    Attributes:
        title: The filename or title of the document
        id: The unique identifier for the document
    """

    title: str
    id: str


class MeetingAttachmentEvent(Event):
    """Event containing a calendar event ID for attachment processing.

    Attributes:
        event_id: The unique identifier of the calendar event
    """

    event_id: str


class AttachmentNameEvent(Event):
    """Event containing an attachment ID for name retrieval.

    Attributes:
        attachment_id: The unique identifier of the document attachment
    """

    attachment_id: str


class AssessTitlesEvent(Event):
    """Event containing attachment metadata for assessment.

    Attributes:
        attachment_id: The unique identifier of the document attachment
        title: The title/filename of the attachment
    """

    attachment_id: str
    title: str


class GetDocContent(Event):
    """Event for triggering document content retrieval.

    Attributes:
        attachment_id: The unique identifier of the document to retrieve
    """

    attachment_id: str


class MeetingNotesWorkflow(Workflow):
    """Workflow for extracting meeting notes from calendar event attachments.

    This workflow processes calendar events to find and extract meeting notes from
    attached Google Documents. It identifies the most likely document containing
    meeting notes and retrieves its content.

    Attributes:
        mcp_client: Client for interacting with Google services via MCP
        llm: Language model for document analysis and classification
    """

    def __init__(self, llm, *args, **kwargs):
        """Initialize the meeting notes workflow.

        Args:
            llm: The language model to use for document analysis
            *args: Additional positional arguments for the parent Workflow
            **kwargs: Additional keyword arguments for the parent Workflow
        """
        if llm is None:
            raise ValueError("LLM instance is required")

        super().__init__(*args, **kwargs)

        try:
            self.mcp_client = BasicMCPClient(str(config.config.google_mcp))
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            raise ConnectionError("Cannot initialize MCP client") from e

        self.llm = llm

    @step
    async def get_meetings_for_date(
        self, event: StartEvent
    ) -> MeetingAttachmentEvent | StopEvent:
        """Retrieve calendar events for a specific date and meeting name.

        Searches for calendar events on the specified date that match the given
        meeting name (case-insensitive).

        Args:
            event: StartEvent containing date and meeting name to search for

        Returns:
            MeetingAttachmentEvent with the event ID if found, or
            StopEvent if no matching events found or an error occurred
        """
        logger.info(f"Getting meeting events for {event.meeting} on {event.date}")

        print(event.date.year)

        try:
            calendar_events = await self.mcp_client.call_tool(
                "get_events_by_date",
                {
                    "year": event.date.year,
                    "day": event.date.day,
                    "month": event.date.month,
                },
            )
        except ConnectionError as e:
            logger.error(f"Failed to connect to calendar service: {e}")
            return StopEvent(result="connection_error")
        except Exception as e:
            logger.error(f"Error retrieving calendar events: {e}")
            return StopEvent(result="calendar_error")

        try:
            if (
                not hasattr(calendar_events, "structuredContent")
                or "result" not in calendar_events.structuredContent
            ):
                logger.error("Invalid calendar events response format")
                return StopEvent(result="invalid_response")

            events_list = calendar_events.structuredContent["result"]
            if not isinstance(events_list, list):
                logger.error("Calendar events result is not a list")
                return StopEvent(result="invalid_response")

            events_ids = [
                item["id"]
                for item in events_list
                if isinstance(item, dict)
                and "id" in item
                and "summary" in item
                and event.meeting.lower() in item["summary"].lower()
            ]

            if not events_ids:
                logger.warning(f"No matching events found for '{event.meeting}'")
                return StopEvent(result="no_events_found")

            logger.info(f"Found {len(events_ids)} matching events")
            return MeetingAttachmentEvent(event_id=events_ids[0])
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing calendar events response: {e}")
            return StopEvent(result="parse_error")
        except Exception as e:
            logger.error(f"Unexpected error processing calendar events: {e}")
            return StopEvent(result="processing_error")

    @step
    async def get_meeting_attachments_ids(
        self, ctx: Context, event: MeetingAttachmentEvent
    ) -> StopEvent | AttachmentNameEvent | None:
        """Retrieve Google Doc attachment IDs from a calendar event.

        Fetches all Google Document attachments associated with the given
        calendar event and creates events for each attachment.

        Args:
            ctx: Workflow context for storing and passing data
            event: Event containing the calendar event ID

        Returns:
            None if successful (events are sent to context), or StopEvent
            if no attachments found or an error occurred
        """
        try:
            res = await self.mcp_client.call_tool(
                "get_event_gdoc_attachments_ids", {"event_id": event.event_id}
            )
        except ConnectionError as e:
            logger.error(f"Failed to connect to attachments service: {e}")
            return StopEvent(result="connection_error")
        except Exception as e:
            logger.error(f"Error calling attachments service: {e}")
            return StopEvent(result="service_error")

        try:
            if (
                not hasattr(res, "structuredContent")
                or "result" not in res.structuredContent
            ):
                logger.error("Invalid attachments response format")
                return StopEvent(result="invalid_response")

            attachments_ids = res.structuredContent["result"]

            if not isinstance(attachments_ids, list):
                logger.warning(
                    f"Unexpected attachments format: {type(attachments_ids)}"
                )
                return StopEvent(result="invalid_format")

            if not attachments_ids:
                logger.info("No attachments found for this meeting")
                return StopEvent(result="no_attachments")

            logger.info(f"Found {len(attachments_ids)} meeting attachments")
            await ctx.store.set("attachments_ids_len", len(attachments_ids))

            for attachment in attachments_ids:
                if attachment:  # Skip empty attachment IDs
                    ctx.send_event(AttachmentNameEvent(attachment_id=attachment))
            return None

        except Exception as e:
            logger.error(f"Error processing attachments response: {e}")
            return StopEvent(result="processing_error")

    @step
    async def get_attachments_title(
        self, event: AttachmentNameEvent
    ) -> AssessTitlesEvent:
        """Retrieve the title of a Google Document attachment.

        Fetches the document title for the given attachment ID to help
        identify which document contains meeting notes.

        Args:
            event: Event containing the attachment ID

        Returns:
            AssessTitlesEvent containing the attachment ID and its title
        """
        try:
            title = await self.mcp_client.call_tool(
                "get_google_doc_title", {"document_id": event.attachment_id}
            )
        except ConnectionError as e:
            logger.error(
                f"Failed to connect to document service for {event.attachment_id}: {e}"
            )
            # Use fallback title and continue workflow
            return AssessTitlesEvent(
                attachment_id=event.attachment_id,
                title=f"Document {event.attachment_id} (Connection Error)",
            )
        except Exception as e:
            logger.error(
                f"Error calling document service for {event.attachment_id}: {e}"
            )
            # Use fallback title and continue workflow
            return AssessTitlesEvent(
                attachment_id=event.attachment_id,
                title=f"Document {event.attachment_id} (Error)",
            )

        try:
            if (
                not hasattr(title, "structuredContent")
                or "result" not in title.structuredContent
            ):
                logger.warning(f"Invalid title response for {event.attachment_id}")
                doc_title = f"Document {event.attachment_id} (Invalid Response)"
            else:
                doc_title = title.structuredContent["result"]

            if not isinstance(doc_title, str) or not doc_title.strip():
                doc_title = f"Untitled Document ({event.attachment_id})"

            return AssessTitlesEvent(
                attachment_id=event.attachment_id,
                title=doc_title,
            )

        except Exception as e:
            logger.error(
                f"Error processing title response for {event.attachment_id}: {e}"
            )
            # Use fallback title and continue workflow
            return AssessTitlesEvent(
                attachment_id=event.attachment_id,
                title=f"Document {event.attachment_id} (Processing Error)",
            )

    @step
    async def assess_attachments(
        self, event: AssessTitlesEvent, ctx: Context
    ) -> GetDocContent | None:
        """Assess attachment titles to identify the meeting notes document.

        Collects all attachment titles and uses an LLM to identify which
        document is most likely to contain meeting notes based on filename
        patterns and keywords.

        Args:
            event: Event containing attachment metadata
            ctx: Workflow context for collecting events

        Returns:
            GetDocContent event with the identified document ID, or None
            if collection is not complete
        """
        num_to_collect = await ctx.store.get("attachments_ids_len")
        if num_to_collect is None or num_to_collect <= 0:
            logger.error("Invalid number of attachments to collect")
            return None

        results = ctx.collect_events(event, [AssessTitlesEvent] * num_to_collect)
        if results is None:
            return None

        if not results:
            logger.error("No attachment title events collected")
            return None

        logger.info(f"Assessing {len(results)} attachments for meeting notes")

        # Validate all results have required attributes
        valid_results = [
            ev
            for ev in results
            if hasattr(ev, "title")
            and hasattr(ev, "attachment_id")
            and ev.title
            and ev.attachment_id
        ]

        if not valid_results:
            logger.error("No valid attachment events found")
            return None

        files_mapping = {ev.title: ev.attachment_id for ev in valid_results}
        logger.debug(f"Files mapping: {list(files_mapping.keys())}")

        # Fallback: if only one attachment, use it directly
        if len(files_mapping) == 1:
            attachment_id = list(files_mapping.values())[0]
            logger.info("Only one attachment found, using it directly")
            return GetDocContent(attachment_id=attachment_id)

        try:
            langfuse_client = get_langfuse_client()
            session_id = f"meeting-note-workflow-{uuid4()}"

            with langfuse_client.start_as_current_span(name=session_id) as span:
                program = LLMTextCompletionProgram.from_defaults(
                    llm=self.llm,
                    output_cls=FileToId,
                    verbose=True,
                    prompt=IDENTIFY_MEETING_NOTES,
                )

                result = await program.acall(files=files_mapping)

                # Validate that the returned ID exists in our mapping
                if not hasattr(result, "id") or result.id not in files_mapping.values():
                    logger.warning(
                        "LLM returned invalid document ID: "
                        f"{getattr(result, 'id', 'None')}"
                    )
                    # Fallback to first document
                    attachment_id = list(files_mapping.values())[0]
                    logger.info(f"Using fallback document: {attachment_id}")
                else:
                    attachment_id = result.id

                span.update_trace(
                    session_id=session_id,
                    input=IDENTIFY_MEETING_NOTES.format(files=files_mapping),
                    output=str(result),
                )

            langfuse_client.flush()
            return GetDocContent(attachment_id=attachment_id)

        except Exception as e:
            logger.error(f"Error during LLM assessment: {e}")
            # Fallback to first document if LLM fails
            attachment_id = list(files_mapping.values())[0]
            logger.info(f"LLM failed, using fallback document: {attachment_id}")
            return GetDocContent(attachment_id=attachment_id)

    @step
    async def get_doc_content(self, event: GetDocContent) -> StopEvent:
        """Retrieve the content of the identified meeting notes document.

        Fetches the full text content of the Google Document identified
        as containing meeting notes.

        Args:
            event: Event containing the document ID to retrieve

        Returns:
            StopEvent containing the document content
        """
        try:
            doc = await self.mcp_client.call_tool(
                "fetch_google_doc_content", {"document_id": event.attachment_id}
            )
        except ConnectionError as e:
            logger.error(f"Failed to connect to document service: {e}")
            return StopEvent(result="connection_error", error=True)
        except Exception as e:
            logger.error(f"Error calling document service: {e}")
            return StopEvent(result="service_error", error=True)

        try:
            if (
                not hasattr(doc, "structuredContent")
                or "result" not in doc.structuredContent
            ):
                logger.error("Invalid document content response format")
                return StopEvent(result="invalid_response", error=True)

            content = doc.structuredContent["result"]

            if not isinstance(content, str):
                if content is None:
                    logger.warning(
                        f"Document {event.attachment_id} returned null content"
                    )
                    content = ""
                else:
                    logger.info(
                        f"Converting non-string content to string: {type(content)}"
                    )
                    content = str(content)

            if not content.strip():
                logger.warning(f"Document {event.attachment_id} content is empty")
                return StopEvent(result="empty_document", error=False)

            logger.info(
                f"Successfully retrieved document content ({len(content)} chars)"
            )
            return StopEvent(result=content)

        except Exception as e:
            logger.error(f"Error processing document content: {e}")
            return StopEvent(result="processing_error", error=True)
