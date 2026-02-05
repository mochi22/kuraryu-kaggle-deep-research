"""Deep Research Agent implementation."""

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ..config import Settings
from ..tools import SearchTools
from .state import ResearchState

MAX_ITERATIONS = 3


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
        """Generate subqueries from main query or gaps."""
        iteration = state.get("iteration", 0)
        
        if iteration == 0:
            print("\n🤔 ステップ 1: サブクエリを生成中...")
            target = state["query"]
        else:
            print(f"\n🔄 反復 {iteration + 1}: 不足情報を補完するクエリを生成中...")
            target = "不足している観点:\n" + "\n".join(state.get("gaps", []))

        prompt = f"""Research Query: "{state['query']}"

{target}

この質問に包括的に答えるための、3-5個の具体的なSub Queryを生成してください。
Sub Queryのみを1行ずつ返してください。"""

        response = self.llm.invoke([SystemMessage(content="あなたはResearch Assistantです。"), HumanMessage(content=prompt)])
        subqueries = [q.strip() for q in response.content.split("\n") if q.strip() and not q.startswith("#")]
        print(f"✓ {len(subqueries)}個のサブクエリを生成しました")
        
        existing = state.get("subqueries", [])
        return {"subqueries": existing + subqueries, "iteration": iteration + 1}

    def _search_sources(self, state: ResearchState) -> ResearchState:
        """Search multiple sources for information."""
        iteration = state.get("iteration", 1)
        print(f"\n🔍 検索中 (反復 {iteration}/{MAX_ITERATIONS})...")
        
        results = list(state.get("search_results", []))
        new_queries = state["subqueries"][-5:]  # 最新のサブクエリのみ検索
        
        for i, subquery in enumerate(new_queries, 1):
            print(f"  [{i}/{len(new_queries)}] {subquery}")
            arxiv_results = self.search_tools.search_arxiv(subquery, max_results=3)
            web_results = self.search_tools.search_web(subquery, max_results=3)
            kaggle_comps = self.search_tools.search_kaggle_competitions(subquery)
            kaggle_datasets = self.search_tools.search_kaggle_datasets(subquery)

            results.extend([{"query": subquery, "source": "arxiv", **r} for r in arxiv_results])
            results.extend([{"query": subquery, "source": "web", **r} for r in web_results])
            results.extend([{"query": subquery, "source": "kaggle-competition", **r} for r in kaggle_comps])
            results.extend([{"query": subquery, "source": "kaggle-dataset", **r} for r in kaggle_datasets])

        print(f"✓ 合計 {len(results)}個のソースを収集")
        return {"search_results": results}

    def _evaluate_coverage(self, state: ResearchState) -> ResearchState:
        """Evaluate if collected information is sufficient."""
        print("\n📊 情報の網羅性を評価中...")
        
        results_summary = "\n".join([f"- [{r['source']}] {r['title']}" for r in state["search_results"][:30]])
        
        prompt = f"""クエリ: "{state['query']}"

収集した情報源:
{results_summary}

この情報で元のクエリに十分答えられますか？
- 十分な場合: 「SUFFICIENT」とだけ回答
- 不足がある場合: 不足している観点を箇条書きで列挙（最大3つ）"""

        response = self.llm.invoke([SystemMessage(content="あなたはResearch評価者です。"), HumanMessage(content=prompt)])
        
        if "SUFFICIENT" in response.content:
            print("✓ 情報は十分です")
            return {"needs_more_search": False, "gaps": []}
        
        gaps = [line.strip().lstrip("-•").strip() for line in response.content.split("\n") if line.strip() and not line.startswith("不足")]
        gaps = [g for g in gaps if g][:3]
        print(f"⚠ 不足している観点: {len(gaps)}個")
        for gap in gaps:
            print(f"  - {gap}")
        return {"needs_more_search": True, "gaps": gaps}

    def _should_continue_search(self, state: ResearchState) -> str:
        """Decide whether to continue searching or proceed to outline."""
        if state.get("needs_more_search") and state.get("iteration", 0) < MAX_ITERATIONS:
            return "generate_subqueries"
        return "generate_outline"

    def _generate_outline(self, state: ResearchState) -> ResearchState:
        """Generate article outline from search results."""
        print("\n📋 記事のアウトラインを生成中...")
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
        print("\n📝 最終記事を生成中...")
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
        workflow.add_node("evaluate_coverage", self._evaluate_coverage)
        workflow.add_node("generate_outline", self._generate_outline)
        workflow.add_node("generate_article", self._generate_article)

        workflow.set_entry_point("generate_subqueries")
        workflow.add_edge("generate_subqueries", "search_sources")
        workflow.add_edge("search_sources", "evaluate_coverage")
        workflow.add_conditional_edges("evaluate_coverage", self._should_continue_search)
        workflow.add_edge("generate_outline", "generate_article")
        workflow.add_edge("generate_article", END)

        return workflow.compile()

    def research(self, query: str) -> dict:
        """Run research workflow."""
        initial_state: ResearchState = {
            "query": query,
            "subqueries": [],
            "outline": "",
            "search_results": [],
            "article": "",
            "messages": [],
            "iteration": 0,
            "needs_more_search": False,
            "gaps": [],
        }
        result = self.graph.invoke(initial_state)
        return result
