"""Nodo 4 — Recommendation: sempre 1-3 azioni concrete.

1) Consigli a regole, sempre calcolati: priorità allarmi > tendenze in peggioramento > mantenimento.
2) Se LLM_ENABLED=true, un LLM locale li riformula e li collega al contesto (meteo, note).
   Se l'LLM fallisce o l'output non è valido, restano quelli a regole.
Guardrail: solo consigli di stile di vita, nessuna diagnosi o indicazione medica.
"""
from .. import llm, running
from ..state import HealthState

ALERT = {
    "sleep_hours": "Stasera punta ad andare a letto 30-45 min prima del solito.",
    "sleep_score": "Sonno di bassa qualità: stasera niente schermi nell'ultima mezz'ora.",
    "hrv_last_night": "Recupero sotto la tua media: oggi meglio attività leggera che allenamento intenso.",
    "resting_hr": "FC a riposo più alta del solito: idratati bene e tieni leggera l'intensità oggi.",
    "body_battery_at_wake": "Poca carica al risveglio: se puoi, sposta gli impegni più pesanti al pomeriggio.",
    "stress_avg": "Ieri stress alto: oggi inserisci 2 pause di 5 minuti lontano dallo schermo.",
    "steps": "Ieri poco movimento: oggi punta a una camminata di 20-30 minuti.",
}


def sleep_floor_text(ins: dict) -> str:
    """Un solo consiglio sul sonno, con il tono proporzionato a quante notti sono sotto soglia."""
    n, tot, floor = ins["nights_below"], ins["nights_total"], ins["floor"]
    if n >= 3:
        return (f"Sono {n} notti su {tot} sotto le {floor:g} ore: stasera anticipa di 30 minuti "
                f"e tieni fisso l'orario di sveglia per qualche giorno.")
    return f"Stanotte hai dormito meno di {floor:g} ore: stasera punta ad andare a letto 30-45 minuti prima del solito."
TREND = {
    "sleep_hours": "Il sonno sta calando da giorni: fissa un orario per andare a letto e rispettalo per 3 sere.",
    "sleep_score": "Qualità del sonno in calo: prova a tenere costante l'orario di sveglia, anche nel weekend.",
    "hrv_last_night": "HRV in discesa da giorni: alleggerisci il carico di allenamento per qualche giorno.",
    "resting_hr": "FC a riposo in salita: più recupero questa settimana, meno alcol e caffeina la sera.",
    "body_battery_at_wake": "Ti svegli sempre più scarico: anticipa di un po' l'ora in cui vai a dormire.",
    "stress_avg": "Stress in aumento: blocca in agenda uno spazio senza impegni nei prossimi giorni.",
    "steps": "Movimento in calo: aggiungi una camminata fissa alla tua routine quotidiana.",
}
MAINTAIN = {
    "sleep_hours": "Tutto in ordine: il sonno è il punto da proteggere, stasera mantieni il tuo orario.",
    "sleep_score": "Tutto in ordine: per tenere alta la qualità del sonno, cena leggera stasera.",
    "hrv_last_night": "Tutto in ordine: se ti alleni oggi, alterna bene intensità e recupero.",
    "resting_hr": "Tutto in ordine: tieni d'occhio idratazione e caffeina nel pomeriggio.",
    "body_battery_at_wake": "Tutto in ordine: una breve pausa a metà giornata aiuta a mantenere la carica.",
    "stress_avg": "Tutto in ordine: lo stress è il valore più vicino al limite, prenditi una pausa vera oggi.",
    "steps": "Tutto in ordine: sui passi hai margine, prova a fare qualche tragitto a piedi.",
}
GENERIC = "Tutto in ordine: mantieni la routine di ieri, sta funzionando."
LEARNING = "Sto ancora imparando la tua norma: nei prossimi giorni i consigli diventeranno personalizzati."


def rule_recommendations(state: HealthState) -> list[str]:
    recs = []
    insights = state.get("insights", [])
    floor = next((i for i in insights if i["type"] == "sleep_floor"), None)
    if floor:
        recs.append(sleep_floor_text(floor))
    for ins in insights:
        # la soglia assoluta copre già la durata del sonno: niente secondo consiglio sullo stesso tema
        if ins["type"] == "deviation" and not (floor and ins["metric"] == "sleep_hours"):
            recs.append(ALERT.get(ins["metric"]))
    for t in state.get("trends", []):
        if not t["improving"] and not (floor and t["metric"] == "sleep_hours"):
            recs.append(TREND.get(t["metric"]))
    recs += running.recommendations(state.get("running"), state)
    recs = list(dict.fromkeys(r for r in recs if r))       # dedup mantenendo l'ordine
    if not recs:
        if state.get("focus"):
            recs = [MAINTAIN.get(state["focus"], GENERIC)]
        elif any(i["type"] == "baseline" for i in state.get("insights", [])):
            recs = [LEARNING]
        else:
            recs = [GENERIC]
    return recs[:3]


def recommendation_node(state: HealthState) -> dict:
    rules = rule_recommendations(state)
    if not llm.enabled() or not state.get("daily_record"):
        return {"recommendations": rules, "recommendations_source": "regole"}

    result, err = llm.llm_recommendations(state, rules)
    if result:
        return {"recommendations": result["recommendations"], "day_summary": result["summary"],
                "recommendations_source": "llm"}
    return {"recommendations": rules, "recommendations_source": "regole",
            "errors": [f"llm (uso le regole): {err}"]}
