"""CLI interface for Deep Research Agent."""

import sys
from datetime import datetime
from pathlib import Path

from kuraryu_deep_research import DeepResearchAgent, Settings


def main() -> None:
    """Run Deep Research Agent CLI."""
    if len(sys.argv) < 2:
        print("Usage: deep-research <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    settings = Settings()
    agent = DeepResearchAgent(settings)

    print("\n" + "=" * 80)
    print(f"🔍 Deep Research Agent")
    print("=" * 80)
    print(f"\n📌 クエリ: {query}")
    print(f"⏰ 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 80)

    result = agent.research(query)

    print("\n" + "=" * 80)
    print("📊 リサーチ結果")
    print("=" * 80)

    print("\n📝 生成されたサブクエリ:")
    for i, sq in enumerate(result["subqueries"], 1):
        print(f"  {i}. {sq}")

    print(f"\n🔄 検索反復回数: {result.get('iteration', 1)}回")

    print(f"\n📚 収集したソース: {len(result['search_results'])}個")
    source_counts = {}
    for r in result["search_results"]:
        source = r["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    for source, count in source_counts.items():
        print(f"  - {source}: {count}個")

    print("\n📋 記事アウトライン:")
    print("-" * 80)
    print(result["outline"])

    print("\n📄 最終記事:")
    print("=" * 80)
    print(result["article"])
    print("=" * 80)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{timestamp}.md"
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# Research Report: {query}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Subqueries\n\n")
        for i, sq in enumerate(result["subqueries"], 1):
            f.write(f"{i}. {sq}\n")
        f.write(f"\n## Sources\n\n")
        f.write(f"Total: {len(result['search_results'])} sources\n\n")
        for source, count in source_counts.items():
            f.write(f"- {source}: {count}\n")
        f.write("\n## Outline\n\n")
        f.write(result["outline"])
        f.write("\n\n## Article\n\n")
        f.write(result["article"])

    print(f"\n💾 レポート保存先: {output_path.absolute()}")
    print(f"⏰ 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
