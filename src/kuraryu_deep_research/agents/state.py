"""State definitions for Deep Research Agent."""

from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class ResearchState(TypedDict):
    """State for research workflow."""

    query: str
    subqueries: list[str]
    outline: str
    search_results: list[dict]
    article: str
    messages: Annotated[list, add_messages]
    iteration: int
    needs_more_search: bool
    gaps: list[str]
    verification_report: str
