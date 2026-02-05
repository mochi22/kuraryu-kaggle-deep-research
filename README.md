# Deep Research Agent

LangGraphベースのDeep Researchエージェント（AWS Bedrock使用）

## 概要

Deep Researchエージェントは、科学論文や技術文書の自動調査・レビューを行うAIエージェントシステムです。ユーザーのクエリから自動的にサブクエリを生成し、複数のソースから情報を収集して、引用付きの包括的なレポートを生成します。

### 主な特徴

- **反復的な深掘り検索**: 情報が不足していれば自動で追加検索（最大3回）
- **情報の検証・クロスチェック**: 複数ソース間の矛盾検出と信頼性評価
- **検索クエリの動的改善**: 結果が少ないクエリを自動で言い換え
- **深さ制御**: 重要な論文の参照先を再帰的に探索（最大深度2）
- **マルチソース検索**: arXiv、Web（DuckDuckGo）、Kaggle
- **引用管理**: 全ての情報に出典を明示

## ワークフロー

```
┌─────────────────┐
│ クエリ入力      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ サブクエリ生成  │◄─────────────────┐
└────────┬────────┘                  │
         ▼                           │
┌─────────────────┐                  │
│ マルチソース検索 │                  │
│ + クエリ改善    │                  │
└────────┬────────┘                  │
         ▼                           │
┌─────────────────┐    不足あり      │
│ 網羅性評価      │──────────────────┘
└────────┬────────┘
         │ 十分
         ▼
┌─────────────────┐
│ 深掘り調査      │◄────┐
│ (関連論文探索)  │     │ 深度 < MAX
└────────┬────────┘─────┘
         │ 完了
         ▼
┌─────────────────┐
│ 情報検証        │
│ (矛盾検出)      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ アウトライン生成 │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 記事生成        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ レポート出力    │
└─────────────────┘
```

## アーキテクチャ

```
kuraryu_deep_research/
├── agents/
│   ├── research.py      # DeepResearchAgent（メインワークフロー）
│   └── state.py         # ResearchState（状態管理）
├── tools/
│   ├── search.py        # SearchTools（arXiv、DuckDuckGo）
│   └── kaggle.py        # KaggleSearch（Kaggle API）
├── config.py            # Settings（設定管理）
└── cli.py               # CLIインターフェース
```

### 技術スタック

- **LLM**: AWS Bedrock Claude Opus 4.5
- **フレームワーク**: LangGraph（状態管理・ワークフロー）
- **検索**: arXiv API、DuckDuckGo Search、Kaggle API
- **設定管理**: Pydantic Settings

## 機能詳細

### 1. 反復的な深掘り検索
- 収集した情報の網羅性をLLMが評価
- 不足している観点を特定し、追加クエリを生成
- 最大3回まで反復

### 2. 情報の検証・クロスチェック
- 異なるソース間の矛盾を検出
- 信頼性評価（arxiv: 高、Web: 要注意）
- 情報の鮮度チェック

### 3. 検索クエリの動的改善
- 結果が2件未満のクエリを自動検出
- LLMが言い換えを生成（専門用語→一般語、英語キーワード追加等）
- 改善クエリで再検索

### 4. 深さ制御（再帰的探索）
- 重要な論文をLLMが選定
- 選んだ論文の関連研究を検索
- 最大深度2まで再帰的に探索

## セットアップ

```bash
# 依存関係インストール
uv sync

# AWS認証設定
aws configure
# または環境変数: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# Kaggle API認証設定（オプション）
mkdir -p ~/.kaggle
# kaggle.json を ~/.kaggle/ に配置
chmod 600 ~/.kaggle/kaggle.json
```

## 使用方法

```bash
# CLIから実行
uv run deep-research "your research query"

# または直接実行
uv run python -m kuraryu_deep_research.cli "your research query"
```

### 実行例

```bash
uv run deep-research "deep learning for tabular data"
```

### 出力例

```
================================================================================
🔍 Deep Research Agent
================================================================================

📌 クエリ: deep learning for tabular data
⏰ 開始時刻: 2026-02-05 15:00:00

================================================================================

🤔 ステップ 1: サブクエリを生成中...
✓ 4個のサブクエリを生成しました

🔍 検索中 (反復 1/3)...
  [1/4] deep learning architectures for tabular data
  [2/4] comparison with traditional ML methods
  ...
✓ 合計 24個のソースを収集

📊 情報の網羅性を評価中...
✓ 情報は十分です

🔬 深掘り調査中 (深度 1/2)...
  → TabNet: Attentive Interpretable Tabular Learning...
  ✓ 5個の関連論文を発見

🔍 情報の検証・クロスチェック中...
✓ 検証完了

📋 記事のアウトラインを生成中...
✓ アウトラインを生成しました

📝 最終記事を生成中...
✓ 記事を生成しました

================================================================================
📊 リサーチ結果
================================================================================

📝 生成されたサブクエリ:
  1. deep learning architectures for tabular data
  2. comparison with traditional ML methods
  ...

🔄 検索反復回数: 1回

📚 収集したソース: 29個
  - arxiv: 15個
  - web: 8個
  - arxiv-deep: 5個
  - kaggle-competition: 1個

💾 レポート保存先: /path/to/reports/research_report_20260205_150000.md
```

## 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `AWS_REGION` | AWSリージョン | us-west-2 |
| `KAGGLE_USERNAME` | Kaggleユーザー名 | - |
| `KAGGLE_KEY` | Kaggle APIキー | - |

## 設定

`config.py` で以下を調整可能：

```python
aws_region: str = "us-west-2"
model_id: str = "global.anthropic.claude-opus-4-5-20251101-v1:0"
temperature: float = 0.0
max_tokens: int = 4096
```

`research.py` の定数：

```python
MAX_ITERATIONS = 3  # 反復検索の最大回数
MAX_DEPTH = 2       # 深掘り調査の最大深度
```
