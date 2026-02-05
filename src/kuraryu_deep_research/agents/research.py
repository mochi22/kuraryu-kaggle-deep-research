"""Deep Research Agent implementation."""

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ..config import Settings
from ..tools import SearchTools
from .state import ResearchState

MAX_ITERATIONS = 3
MAX_DEPTH = 2


class DeepResearchAgent:
    """Deep Research Agent using LangGraph."""

    def __init__(self, settings: Settings) -> None:
        """Initialize agent."""
        self.settings = settings
        boto_config = Config(read_timeout=6000, retries={"max_attempts": 3})
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            config=boto_config,
        )
        self.llm = ChatBedrock(
            model_id=settings.model_id,
            client=bedrock_client,
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
        new_queries = state["subqueries"][-5:]
        low_result_queries = []

        for i, subquery in enumerate(new_queries, 1):
            print(f"  [{i}/{len(new_queries)}] {subquery}")
            arxiv_results = self.search_tools.search_arxiv(subquery, max_results=3)
            web_results = self.search_tools.search_web(subquery, max_results=3)
            kaggle_comps = self.search_tools.search_kaggle_competitions(subquery)
            kaggle_datasets = self.search_tools.search_kaggle_datasets(subquery)

            query_results = []
            query_results.extend([{"query": subquery, "source": "arxiv", **r} for r in arxiv_results])
            query_results.extend([{"query": subquery, "source": "web", **r} for r in web_results])
            query_results.extend([{"query": subquery, "source": "kaggle-competition", **r} for r in kaggle_comps])
            query_results.extend([{"query": subquery, "source": "kaggle-dataset", **r} for r in kaggle_datasets])

            if len(query_results) < 2:
                low_result_queries.append(subquery)
            results.extend(query_results)

        # 結果が少ないクエリを改善して再検索
        if low_result_queries:
            print(f"\n🔧 {len(low_result_queries)}個のクエリを改善中...")
            improved = self._improve_queries(low_result_queries, state["query"])
            for subquery in improved:
                print(f"  → {subquery}")
                arxiv_results = self.search_tools.search_arxiv(subquery, max_results=3)
                web_results = self.search_tools.search_web(subquery, max_results=3)
                results.extend([{"query": subquery, "source": "arxiv", **r} for r in arxiv_results])
                results.extend([{"query": subquery, "source": "web", **r} for r in web_results])

        print(f"✓ 合計 {len(results)}個のソースを収集")
        return {"search_results": results}

    def _improve_queries(self, queries: list[str], original_query: str) -> list[str]:
        """Improve queries that returned few results."""
        prompt = f"""元のクエリ: "{original_query}"

以下の検索クエリは結果が少なかったです:
{chr(10).join(f'- {q}' for q in queries)}

各クエリを言い換えて、より検索結果が得られやすい形に改善してください。
- 専門用語を一般的な言葉に
- 英語のキーワードを追加
- より広い概念に変更

改善したクエリのみを1行ずつ出力してください。"""

        response = self.llm.invoke([
            SystemMessage(content="あなたは検索クエリ最適化の専門家です。"),
            HumanMessage(content=prompt)
        ])
        return [q.strip().lstrip("-•").strip() for q in response.content.split("\n") if q.strip()][:len(queries)]

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
        """Decide whether to continue searching or proceed to deep dive."""
        if state.get("needs_more_search") and state.get("iteration", 0) < MAX_ITERATIONS:
            return "generate_subqueries"
        return "deep_dive"

    def _deep_dive(self, state: ResearchState) -> ResearchState:
        """Extract key references from results and explore them deeper."""
        depth = state.get("depth", 0)
        if depth >= MAX_DEPTH:
            return {}

        print(f"\n🔬 深掘り調査中 (深度 {depth + 1}/{MAX_DEPTH})...")

        explored = state.get("explored_urls") or set()
        arxiv_results = [r for r in state["search_results"] if r["source"] == "arxiv" and r["url"] not in explored]

        if not arxiv_results:
            print("  深掘り対象なし")
            return {"depth": depth + 1}

        # 重要な論文を特定
        prompt = f"""以下の論文から、元のクエリ「{state['query']}」を深く理解するために
さらに調査すべき最も重要な論文を最大2つ選んでください。

論文リスト:
{chr(10).join(f"- {r['title']}" for r in arxiv_results[:10])}

選んだ論文のタイトルのみを1行ずつ出力してください。"""

        response = self.llm.invoke([
            SystemMessage(content="あなたはResearch Assistantです。"),
            HumanMessage(content=prompt)
        ])

        selected_titles = [t.strip().lstrip("-•").strip() for t in response.content.split("\n") if t.strip()]
        selected = [r for r in arxiv_results if any(t in r["title"] for t in selected_titles)][:2]

        if not selected:
            return {"depth": depth + 1}

        # 選んだ論文の関連研究を検索
        new_results = []
        new_explored = set(explored)
        for paper in selected:
            print(f"  → {paper['title'][:50]}...")
            new_explored.add(paper["url"])

            # 論文タイトルで関連研究を検索
            related = self.search_tools.search_arxiv(paper["title"], max_results=3)
            for r in related:
                if r["url"] not in new_explored:
                    new_results.append({"query": f"related to: {paper['title']}", "source": "arxiv-deep", **r})
                    new_explored.add(r["url"])

        print(f"  ✓ {len(new_results)}個の関連論文を発見")
        return {
            "search_results": state["search_results"] + new_results,
            "depth": depth + 1,
            "explored_urls": new_explored,
        }

    def _should_continue_deep_dive(self, state: ResearchState) -> str:
        """Decide whether to continue deep diving."""
        if state.get("depth", 0) < MAX_DEPTH:
            new_deep_results = [r for r in state["search_results"] if r["source"] == "arxiv-deep"]
            if new_deep_results:
                return "deep_dive"
        return "verify_information"

    def _verify_information(self, state: ResearchState) -> ResearchState:
        """Verify information across sources and detect contradictions."""
        print("\n🔍 情報の検証・クロスチェック中...")

        results_text = "\n\n".join(
            [f"[{i+1}] [{r['source']}] {r['title']}\n{r.get('summary', r.get('content', ''))[:300]}"
             for i, r in enumerate(state["search_results"][:20])]
        )

        prompt = f"""クエリ: "{state['query']}"

収集した情報:
{results_text}

以下の観点で情報を検証してください:
1. 矛盾する主張: 異なるソース間で矛盾する情報があれば指摘
2. 信頼性評価: 学術論文(arxiv)は高信頼、一般Web記事は要注意
3. 情報の鮮度: 古い情報と新しい情報の違いがあれば指摘

検証レポートを簡潔に日本語で出力してください。矛盾がなければ「主要な矛盾は検出されませんでした」と記載。"""

        response = self.llm.invoke([
            SystemMessage(content="あなたは情報検証の専門家です。"),
            HumanMessage(content=prompt)
        ])

        print("✓ 検証完了")
        return {"verification_report": response.content}

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
        
        verification = state.get("verification_report", "")
        verification_note = f"\n\n検証結果を考慮してください:\n{verification}" if verification else ""
        
        prompt = f"""以下のアウトラインに従って、包括的なリサーチ記事を日本語で執筆してください:

{state['outline']}

以下の情報源を使用してください:
{results_text}
{verification_note}

各主張の後に引用URL [source] を含めてください。
矛盾する情報がある場合は両論併記してください。
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
        workflow.add_node("deep_dive", self._deep_dive)
        workflow.add_node("verify_information", self._verify_information)
        workflow.add_node("generate_outline", self._generate_outline)
        workflow.add_node("generate_article", self._generate_article)

        workflow.set_entry_point("generate_subqueries")
        workflow.add_edge("generate_subqueries", "search_sources")
        workflow.add_edge("search_sources", "evaluate_coverage")
        workflow.add_conditional_edges("evaluate_coverage", self._should_continue_search)
        workflow.add_conditional_edges("deep_dive", self._should_continue_deep_dive)
        workflow.add_edge("verify_information", "generate_outline")
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
            "verification_report": "",
            "depth": 0,
            "explored_urls": set(),
        }
        result = self.graph.invoke(initial_state)
        return result
