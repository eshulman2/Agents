"""Pydantic models for workflow data structures."""

from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """Single action item extracted from meeting notes."""

    title: str = Field(..., description="Brief title of the action item")
    description: str = Field(
        ..., description="Detailed description of what needs to be done"
    )
    assignee: Optional[str] = Field(
        None, description="Person responsible for this action"
    )
    due_date: Optional[Union[datetime, str]] = Field(
        None, description="When this action should be completed (datetime or 'TBD')"
    )
    priority: str = Field(
        "medium", description="Priority level: low, medium, high, urgent"
    )
    category: str = Field("general", description="Category or type of action item")
    dependencies: List[str] = Field(
        default_factory=list, description="Other action items this depends on"
    )
    estimated_effort: Optional[str] = Field(
        None, description="Estimated time or effort required"
    )


class ActionItemsList(BaseModel):
    """Complete list of action items from meeting notes."""

    meeting_title: str = Field(..., description="Title or subject of the meeting")
    meeting_date: Union[datetime, str] = Field(
        ..., description="Date when the meeting occurred (datetime or 'TBD')"
    )
    action_items: List[ActionItem] = Field(
        ..., description="List of extracted action items"
    )
    summary: Optional[str] = Field(None, description="Brief summary of the meeting")
    participants: List[str] = Field(
        default_factory=list, description="Meeting participants"
    )
    next_meeting_date: Optional[Union[datetime, str]] = Field(
        None, description="Date of next follow-up meeting (datetime or 'TBD')"
    )


class ReviewFeedback(BaseModel):
    """Feedback from review process."""

    requires_changes: bool = Field(..., description="Whether changes are needed")
    feedback: str = Field(
        ..., description="Specific feedback or suggestions for improvement"
    )
    approved_items: List[int] = Field(
        default_factory=list, description="Indices of approved action items"
    )
    rejected_items: List[int] = Field(
        default_factory=list, description="Indices of rejected action items"
    )


class AgentRoutingDecision(BaseModel):
    """Decision for routing action item to specific agent."""

    action_item_index: int = Field(
        ..., description="Index of the action item in the list"
    )
    agent_name: str = Field(..., description="Name of the selected agent")
    routing_reason: str = Field(
        ..., description="Explanation for why this agent was chosen"
    )
    requires_human_approval: bool = Field(
        False, description="Whether human approval is needed"
    )


class AgentExecutionResult(BaseModel):
    """Result from agent execution of an action item."""

    action_item_index: int = Field(
        ..., description="Index of the action item that was processed"
    )
    agent_name: str = Field(
        ..., description="Name of the agent that processed the item"
    )
    success: bool = Field(..., description="Whether the execution was successful")
    result: str = Field(..., description="Result or output from the agent")
    error_message: Optional[str] = Field(
        None, description="Error message if execution failed"
    )
    execution_time: Optional[float] = Field(
        None, description="Time taken for execution in seconds"
    )
