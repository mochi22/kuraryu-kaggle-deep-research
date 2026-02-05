"""Kaggle MCP integration for Deep Research Agent."""

from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleSearch:
    """Kaggle search using official API."""

    def __init__(self) -> None:
        """Initialize Kaggle API client."""
        self.api = KaggleApi()
        try:
            self.api.authenticate()
            self.authenticated = True
        except Exception:
            self.authenticated = False

    def search_competitions(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search Kaggle competitions."""
        if not self.authenticated:
            return []
        try:
            competitions = self.api.competitions_list(search=query)[:max_results]
            return [
                {
                    "title": comp.title,
                    "url": f"https://www.kaggle.com/competitions/{comp.ref}",
                    "content": f"{comp.description or ''} | Deadline: {comp.deadline} | Reward: {comp.reward}",
                }
                for comp in competitions
            ]
        except Exception:
            return []

    def search_datasets(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search Kaggle datasets."""
        if not self.authenticated:
            return []
        try:
            datasets = self.api.dataset_list(search=query, max_size=max_results)
            return [
                {
                    "title": dataset.title,
                    "url": f"https://www.kaggle.com/datasets/{dataset.ref}",
                    "content": f"{dataset.subtitle or ''} | Size: {dataset.totalBytes} bytes | Downloads: {dataset.downloadCount}",
                }
                for dataset in datasets[:max_results]
            ]
        except Exception:
            return []

    def search_notebooks(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search Kaggle notebooks (kernels)."""
        if not self.authenticated:
            return []
        try:
            kernels = self.api.kernels_list(search=query, page_size=max_results)
            return [
                {
                    "title": kernel.title,
                    "url": f"https://www.kaggle.com/code/{kernel.ref}",
                    "content": f"Author: {kernel.author} | Votes: {kernel.totalVotes} | Language: {kernel.language}",
                }
                for kernel in kernels
            ]
        except Exception:
            return []

    def search_discussions(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search Kaggle discussions via web search."""
        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            search_query = f"site:kaggle.com/discussions {query}"
            results = ddgs.text(search_query, max_results=max_results)
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in results]
        except Exception:
            return []
