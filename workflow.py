from langgraph.graph import StateGraph, END

from ai_agent import AppState
from nodes import (
    load_data_node,
    handle_missing_node,
    ml_insights_node,
    plot_node,
    llm_summary_node,
    report_node,
)


def build_graph():
    graph = StateGraph(AppState)

    graph.add_node("load_data", load_data_node)
    graph.add_node("handle_missing", handle_missing_node)
    graph.add_node("ml_insights", ml_insights_node)
    graph.add_node("plot_data", plot_node)
    graph.add_node("llm_summary", llm_summary_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("load_data")

    graph.add_edge("load_data", "handle_missing")
    graph.add_edge("handle_missing", "ml_insights")
    graph.add_edge("ml_insights", "plot_data")
    graph.add_edge("plot_data", "llm_summary")
    graph.add_edge("llm_summary", "report")
    graph.add_edge("report", END)

    return graph.compile()


app = build_graph()
