"""Search tools for Deep Research Agent."""

from typing import Any

import arxiv
from duckduckgo_search import DDGS

from .kaggle import KaggleSearch


class SearchTools:
    """Collection of search tools."""

    def __init__(self) -> None:
        """Initialize search tools."""
        self.ddgs = DDGS()
        self.kaggle = KaggleSearch()

    def search_arxiv(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search arXiv for papers."""
        try:
            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            return [
                {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "url": result.entry_id,
                    "published": result.published.isoformat(),
                }
                for result in client.results(search)
            ]
        except Exception:
            return []

    def search_web(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search web using DuckDuckGo."""
        try:
            results = self.ddgs.text(query, max_results=max_results)
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in results]
        except Exception:
            return []

    def search_kaggle_competitions(self, query: str) -> list[dict[str, Any]]:
        """Search Kaggle competitions."""
        return self.kaggle.search_competitions(query)

    def search_kaggle_datasets(self, query: str) -> list[dict[str, Any]]:
        """Search Kaggle datasets."""
        return self.kaggle.search_datasets(query)

    def search_kaggle_notebooks(self, query: str) -> list[dict[str, Any]]:
        """Search Kaggle notebooks."""
        return self.kaggle.search_notebooks(query)

    def search_kaggle_discussions(self, query: str) -> list[dict[str, Any]]:
        """Search Kaggle discussions."""
        return self.kaggle.search_discussions(query)
