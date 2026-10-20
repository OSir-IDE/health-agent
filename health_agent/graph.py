"""Definizione del grafo LangGraph."""
from langgraph.graph import END, StateGraph

from .nodes import (analysis_node, extras_node, ingestion_node, news_node, normalization_node,
                    notifier_node, recommendation_node, running_node, weekly_node)
from .state import HealthState


def has_data(state: HealthState) -> str:
    """Se non arriva nessun dato, salta l'analisi (ma meteo e frase arrivano comunque)."""
    return "normalization" if state.get("garmin_raw") else "extras"


def build_graph():
    g = StateGraph(HealthState)
    g.add_node("ingestion", ingestion_node)
    g.add_node("normalization", normalization_node)
    g.add_node("analysis", analysis_node)
    g.add_node("running", running_node)
    g.add_node("recommendation", recommendation_node)
    g.add_node("extras", extras_node)
    g.add_node("notifier", notifier_node)
    g.add_node("news", news_node)
    g.add_node("weekly", weekly_node)

    g.set_entry_point("ingestion")
    g.add_conditional_edges("ingestion", has_data,
                            {"normalization": "normalization", "extras": "extras"})
    g.add_edge("normalization", "analysis")
    g.add_edge("analysis", "running")
    g.add_edge("running", "extras")
    g.add_edge("extras", "recommendation")
    g.add_edge("recommendation", "notifier")
    g.add_edge("notifier", "news")
    g.add_edge("news", "weekly")
    g.add_edge("weekly", END)
    return g.compile()
