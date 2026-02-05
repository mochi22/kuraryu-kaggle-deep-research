"""Deep Research Agent implementation."""

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ..config import Settings
from ..tools import SearchTools
from .state import ResearchState


class DeepResearchAgent:
    """Deep Research Agent using LangGraph."""

    def __init__(self, settings: Settings) -> None:
        """Initialize agent."""
        self.settings = settings
        self.llm = ChatBedrock(
            model_id=settings.model_id,
            region_name=settings.aws_region,
            model_kwargs={"temperature": settings.temperature, "max_tokens": settings.max_tokens},
        )
        self.search_tools = SearchTools()
        self.graph = self._build_graph()

    def _generate_subqueries(self, state: ResearchState) -> ResearchState:
        """Generate subqueries from main query."""
        print("\n🤔 ステップ 1/4: サブクエリを生成中...")
        prompt = f"""Research Query: "{state['query']}"

この質問に包括的に答えるための、3-5個の具体的なSub Queryを生成してください。
Sub Queryのみを1行ずつ返してください。"""

        response = self.llm.invoke([SystemMessage(content="あなたはResearch Assistantです。"), HumanMessage(content=prompt)])
        subqueries = [q.strip() for q in response.content.split("\n") if q.strip()]
        print(f"✓ {len(subqueries)}個のサブクエリを生成しました")
        return {"subqueries": subqueries}

    def _search_sources(self, state: ResearchState) -> ResearchState:
        """Search multiple sources for information."""
        print("\n🔍 ステップ 2/4: 複数ソースから情報を検索中...")
        results = []
        for i, subquery in enumerate(state["subqueries"], 1):
            print(f"  [{i}/{len(state['subqueries'])}] {subquery}")
            arxiv_results = self.search_tools.search_arxiv(subquery, max_results=3)
            web_results = self.search_tools.search_web(subquery, max_results=3)
            kaggle_comps = self.search_tools.search_kaggle_competitions(subquery)
            kaggle_datasets = self.search_tools.search_kaggle_datasets(subquery)
            kaggle_notebooks = self.search_tools.search_kaggle_notebooks(subquery)
            kaggle_discussions = self.search_tools.search_kaggle_discussions(subquery)

            results.extend([{"query": subquery, "source": "arxiv", **r} for r in arxiv_results])
            results.extend([{"query": subquery, "source": "web", **r} for r in web_results])
            results.extend([{"query": subquery, "source": "kaggle-competition", **r} for r in kaggle_comps])
            results.extend([{"query": subquery, "source": "kaggle-dataset", **r} for r in kaggle_datasets])
            results.extend([{"query": subquery, "source": "kaggle-notebook", **r} for r in kaggle_notebooks])
            results.extend([{"query": subquery, "source": "kaggle-discussion", **r} for r in kaggle_discussions])
        print(f"✓ {len(results)}個のソースを収集しました")
        return {"search_results": results}

    def _generate_outline(self, state: ResearchState) -> ResearchState:
        """Generate article outline from search results."""
        print("\n📋 ステップ 3/4: 記事のアウトラインを生成中...")
        results_text = "\n\n".join([f"- {r['title']}: {r.get('summary', r.get('content', ''))[:200]}" for r in state["search_results"]])
        prompt = f"""クエリ: "{state['query']}"

検索結果:
{results_text}

これらの情報に基づいて、詳細な記事のアウトラインをセクションとサブセクションで作成してください。
日本語で出力してください。"""

        response = self.llm.invoke([SystemMessage(content="あなたはResearch Writerです。"), HumanMessage(content=prompt)])
        print("✓ アウトラインを生成しました")
        return {"outline": response.content}

    def _generate_article(self, state: ResearchState) -> ResearchState:
        """Generate final article."""
        print("\n📝 ステップ 4/4: 最終記事を生成中...")
        results_text = "\n\n".join([f"[{r['source']}] {r['title']}\n{r.get('summary', r.get('content', ''))}\nURL: {r['url']}" for r in state["search_results"]])
        prompt = f"""以下のアウトラインに従って、包括的なリサーチ記事を日本語で執筆してください:

{state['outline']}

以下の情報源を使用してください:
{results_text}

各主張の後に引用URL [source] を含めてください。
全て日本語で出力してください。"""

        response = self.llm.invoke([SystemMessage(content="あなたはResearch Writerです。"), HumanMessage(content=prompt)])
        print("✓ 記事を生成しました")
        return {"article": response.content}

    def _build_graph(self) -> StateGraph:
        """Build the research workflow graph."""
        workflow = StateGraph(ResearchState)
        workflow.add_node("generate_subqueries", self._generate_subqueries)
        workflow.add_node("search_sources", self._search_sources)
        workflow.add_node("generate_outline", self._generate_outline)
        workflow.add_node("generate_article", self._generate_article)

        workflow.set_entry_point("generate_subqueries")
        workflow.add_edge("generate_subqueries", "search_sources")
        workflow.add_edge("search_sources", "generate_outline")
        workflow.add_edge("generate_outline", "generate_article")
        workflow.add_edge("generate_article", END)

        return workflow.compile()

    def research(self, query: str) -> dict:
        """Run research workflow."""
        initial_state = {"query": query, "subqueries": [], "outline": "", "search_results": [], "article": "", "messages": []}
        result = self.graph.invoke(initial_state)
        return result
