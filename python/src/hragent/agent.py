from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from .analysis import AnalysisResult, analyze_session
from .config import AppConfig, get_report_dir, load_config
from .plots import generate_plots
from .session_store import load_session

MAX_REFLECTIONS = 1


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    analysis: dict[str, Any] | None
    plots: list[str]
    report_markdown: str
    complete: bool
    session_path: str
    report_dir: str
    reflection_count: int
    sample_rate_hz: float
    measurement_time: str
    session_name: str


def _serialize_analysis(result: AnalysisResult) -> dict[str, Any]:
    return {
        "session": {
            "path": str(result.session.path),
            "n_samples": result.session.n_samples,
            "duration_sec": result.session.duration_sec,
            "sample_rate_hz": result.session.sample_rate_hz,
            "start_timestamp_ms": result.session.start_timestamp_ms,
            "end_timestamp_ms": result.session.end_timestamp_ms,
        },
        "hr": {
            "mean_hr_bpm": result.hr.mean_hr_bpm,
            "min_hr_bpm": result.hr.min_hr_bpm,
            "max_hr_bpm": result.hr.max_hr_bpm,
            "time_domain": result.hr.time_domain,
            "freq_domain": result.hr.freq_domain,
            "artifact_percentage": result.hr.artifact_percentage,
            "quality": result.hr.quality,
        },
    }


def _build_prompt(
    analysis: dict[str, Any],
    plots: Sequence[str],
    measurement_time: str,
    session_name: str,
) -> str:
    session = analysis["session"]
    hr = analysis["hr"]
    time_domain = hr.get("time_domain", {})
    freq_domain = hr.get("freq_domain", {})

    lines = [
        "タイトルにはセッションIDと計測日時を含めてください（例: # 心拍レポート - test6 - 2025-11-25 11:39）。",
        f"セッションID: {session_name}",
        f"計測日時: {measurement_time}",
        f"計測時間: {session['duration_sec']:.1f} 秒",
        f"推定サンプリング周波数: {session['sample_rate_hz']:.1f} Hz",
        f"サンプル数: {session['n_samples']}",
        "使用センサ: Pulse Sensor (SEN-11574相当)",
        "",  # separator
        f"平均心拍数: {hr['mean_hr_bpm']:.1f} bpm (最小 {hr['min_hr_bpm']:.1f}, 最大 {hr['max_hr_bpm']:.1f})",
        f"アーチファクト率: {hr['artifact_percentage']:.1f}%、品質判定: {hr['quality']}",
        "時間領域 HRV 指標:",
        f"  - SDNN: {time_domain.get('sdnn', 0):.1f} ms",
        f"  - RMSSD: {time_domain.get('rmssd', 0):.1f} ms",
        f"  - pNN50: {time_domain.get('pnni_50', 0):.1f}%",
        "周波数領域 HRV 指標:",
        f"  - LF: {freq_domain.get('lf', 0):.1f}",
        f"  - HF: {freq_domain.get('hf', 0):.1f}",
        f"  - LF/HF: {freq_domain.get('lf_hf_ratio', 0):.2f}",
        "プロット一覧:",
    ]
    if plots:
        lines.extend(f"  - {plot}" for plot in plots)
    else:
        lines.append("  - プロットは利用できませんでした。")
    lines.append("以上の情報を基に、指定されたMarkdown形式でレポートを作成してください。")
    lines.append("""
出力すべきセクション:
1. ## 概要
2. ## 測定条件
3. ## 心拍解析結果
4. ## HRV解析結果
5. ## 信号品質
6. ## 注意事項
7. ## まとめ（本セッションで読み取れること）
""")
    lines.append(
        "各セクションでは上記のメトリクスを引用し、専門用語には短く意味を添え、値の大小が一般的に示す傾向をやわらかい言葉で説明してください。"
        "解釈はデータの範囲に留め、医療診断を行わず、このシステムが医療機器ではない旨を繰り返し述べてください。"
        "読む人が初めて心拍指標に触れる前提で、読みやすい日本語を使ってください。"
        "必要に応じて一般的な目安（例: 安静時成人の心拍数はおおむね60-100 bpm）と比較し、高低を一言で補足しつつ個人差と非医療であることを強調してください。"
        "「まとめ」では、このセッションで読み取れる状態を2〜4行で具体的に書き、数値に基づいた簡潔な要約を示してください。"
        "ただし診断や断定は避け、体調が気になる場合は休息や専門家相談を推奨する言い回しにしてください。"
    )
    lines.append(
        "プロットを挿入する際は `![説明](plots/ファイル名.png)` 形式を用い、必ず relative path を使用してください。"
    )
    lines.append(
        "「信号品質」では品質が低いときの注意点と、再計測時の簡単な改善提案を含めてください。"
    )
    lines.append(
        "「注意事項」では医療目的ではないことと、気になる場合は専門家に相談するよう促してください。"
    )
    return "\n".join(lines)


def _build_system_prompt() -> str:
    return (
        "あなたは、心拍やHRVに馴染みのない人にも伝わるレポートを書くAIです。"
        "与えられた指標のみを利用し、推測で測定内容を創作しないでください。"
        "専門用語は短く意味を添え、平易で丁寧な日本語を使ってください。"
        "医療的な診断や断定は避け、一般的な傾向に留めます。"
        "必ず「## 注意事項」に医療機器ではないことと診断目的でないことを明記し、専門家相談を促してください。"
        "Markdownのコードフェンス（```）は使わず、指定された見出しで出力してください。"
    )


def _strip_markdown_fence(text: str) -> str:
    """Remove surrounding ```...``` fences if model returns them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove first line fence
        parts = stripped.splitlines()
        # drop first line
        parts = parts[1:]
        # drop trailing fence if present
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        stripped = "\n".join(parts).strip()
    return stripped


def _build_graph(config: AppConfig, model: ChatOpenAI) -> StateGraph[AgentState]:
    graph = StateGraph(state_schema=AgentState)

    def load_and_analyze_node(state: AgentState) -> dict[str, Any]:
        session_path = Path(state["session_path"])
        report_dir = Path(state["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        df, metadata = load_session(session_path)
        analysis_result = analyze_session(df, metadata, state.get("sample_rate_hz", config.sample_rate_hz))
        plot_paths = generate_plots(df, analysis_result, report_dir)
        plots = []
        for path in (plot_paths.signal_plot, plot_paths.hr_plot, plot_paths.hrv_psd_plot):
            if path is None:
                continue
            rel = path.relative_to(report_dir)
            plots.append(rel.as_posix())
        return {
            "analysis": _serialize_analysis(analysis_result),
            "plots": plots,
        }

    def generate_report_node(state: AgentState) -> dict[str, Any]:
        analysis = state.get("analysis")
        if analysis is None:
            raise RuntimeError("分析結果がありません")
        prompt = _build_prompt(
            analysis,
            state.get("plots", []),
            state.get("measurement_time", ""),
            state.get("session_name", "unknown"),
        )
        ai_message = model.invoke(prompt)
        content = ai_message.content if isinstance(ai_message, AIMessage) else str(ai_message)
        response_text = _strip_markdown_fence(content)
        return {
            "report_markdown": response_text,
            "messages": [
                SystemMessage(content=_build_system_prompt()),
                HumanMessage(content=prompt),
                AIMessage(content=response_text),
            ],
            "complete": True,
        }

    def reflection_node(state: AgentState) -> dict[str, Any]:
        current = state.get("report_markdown", "")
        if not current:
            return {}
        reflection_prompt = (
            "既存のレポートを読み直し、表現を磨きつつ過度な断定を避け、信号品質の注意喚起を明記してください。"
            "全体の構成やセクションは維持し、追加の医学的判断を盛り込まないでください。"
            f"---\n{current}"
        )
        ai_message = model.invoke(reflection_prompt)
        content = ai_message.content if isinstance(ai_message, AIMessage) else str(ai_message)
        improved = _strip_markdown_fence(content)
        return {
            "report_markdown": improved,
            "messages": [
                HumanMessage(content=reflection_prompt),
                AIMessage(content=improved),
            ],
            "reflection_count": state.get("reflection_count", 0) + 1,
        }

    def should_reflect(state: AgentState) -> str:
        if state.get("reflection_count", 0) < MAX_REFLECTIONS:
            return "reflection_node"
        return END

    graph.add_node("load_and_analyze_node", load_and_analyze_node)
    graph.add_node("generate_report_node", generate_report_node)
    graph.add_node("reflection_node", reflection_node)
    graph.add_edge("load_and_analyze_node", "generate_report_node")
    graph.add_conditional_edges("generate_report_node", should_reflect)
    graph.set_entry_point("load_and_analyze_node")
    graph.set_finish_point("reflection_node")
    return graph


def generate_report_for_session(
    session_path: Path,
    *,
    model: ChatOpenAI | None = None,
) -> str:
    logger = logging.getLogger(__name__)
    config = load_config()
    session_name = session_path.stem
    report_dir = get_report_dir(config.paths, session_name)
    openai_model = model or ChatOpenAI(model=config.openai_model)
    graph = _build_graph(config, openai_model)
    compiled = graph.compile()
    measurement_time = datetime.utcnow().isoformat()
    logger.info("Analyze session: %s", session_path)
    initial_state: AgentState = {
        "messages": [],
        "analysis": None,
        "plots": [],
        "report_markdown": "",
        "complete": False,
        "session_path": str(session_path),
        "report_dir": str(report_dir),
        "reflection_count": 0,
        "sample_rate_hz": config.sample_rate_hz,
        "measurement_time": measurement_time,
        "session_name": session_name,
    }
    final_state = compiled.invoke(initial_state)
    logger.info("Report generation completed")
    report = final_state.get("report_markdown")
    if not report:
        raise RuntimeError("レポートの生成に失敗しました")
    return report
