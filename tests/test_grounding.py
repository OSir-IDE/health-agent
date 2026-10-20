"""Controlli di ancoraggio: casi reali del giornale (esempi dei tuoi report) e casi limite. Nessuna rete."""
import json

import pytest

from health_agent import grounding, llm


# ------------------------------------------------------------------ termini delicati e direzione
def test_nobel_not_in_sources_is_blocked_and_in_sources_is_ok():
    sent = "Navi Pillay, ex giudice della Corte, ha ricevuto il premio Nobel per la pace"
    assert grounding.ungrounded(sent, "Sanzioni Usa alla Corte penale internazionale. Navi Pillay, ex giudice")
    assert not grounding.ungrounded(sent, "Navi Pillay, ex giudice della Corte, riceve il Nobel per la pace")


def test_coma_vs_terapia_intensiva():
    sent = "Davide Iacopini rimane in coma dopo un'aggressione"
    assert grounding.ungrounded(sent, "L'attore Iacopini aggredito a Roma, ancora in terapia intensiva")
    assert not grounding.ungrounded(sent, "L'attore Iacopini è in coma dopo l'aggressione")


def test_english_sources_support_italian_sentences():
    assert not grounding.ungrounded("Almeno 30 persone sono morte nell'attacco", "Russian attack on two buses kills 30")
    assert grounding.ungrounded("Ci sono stati arresti", "Russian attack on two buses kills 30")


def test_direction_flip_is_blocked_but_mixed_sources_pass():
    assert grounding.ungrounded("La Borsa di Milano chiude in rialzo", "Borsa: Milano chiude in calo con le banche")
    assert not grounding.ungrounded("La Borsa di Milano chiude in rialzo",
                                    "Milano chiude in rialzo, ma le banche calano")
    assert grounding.ungrounded("Lo spread scende", "Salgono spread e rendimenti BTP")


def test_innocent_sentence_passes():
    assert not grounding.ungrounded("Verstappen ottiene la pole nella gara breve di Singapore",
                                    "F1, Gp Singapore: Verstappen in pole nella Sprint Race")


def test_known_limit_role_inversion_is_not_detected():
    # documentato: questo controllo NON vede lo scambio dei ruoli (serve il giudice sulle frasi)
    assert not grounding.ungrounded("Aggredito a Roma da una banda in monopattino",
                                    "aggredito sul monopattino dalla banda della minicar")


# ------------------------------------------------------------------ nomi propri (solo registro)
def test_unseen_names_ignores_sentence_start_and_common_words():
    src = "Sanzioni Usa alla Corte penale internazionale. Ex giudice Navi Pillay"
    assert grounding.unseen_names("Gli Stati Uniti sanzionano la Corte, mentre Pillay parla", src) == []
    assert grounding.unseen_names("Intanto Netanyahu ha risposto a Pillay", src) == ["Netanyahu"]


# ------------------------------------------------------------------ integrazione nei validatori
STORIES = ["Borsa: Milano chiude in calo. Banche in rosso", "Premio Nobel per la pace a Maria Rossi, attivista"]
FILLER = " Quattro parole in più per arrivare alla lunghezza minima del testo richiesto dal controllo di forma [1]."


def _text(*sents):
    return " ".join(sents) + FILLER * 3


def test_validate_narrative_checks_each_sentence_against_cited_sources_only():
    ok = _text("La Borsa di Milano chiude in calo [1].", "Il premio Nobel per la pace va a Maria Rossi [2].")
    assert llm.validate_narrative(ok, 2, STORIES)
    # "Nobel" è nelle fonti, ma della storia 2: la frase cita la 1 -> bloccata
    bad = _text("La Borsa chiude in calo [1].", "Il premio Nobel è andato a un attivista [1].")
    with pytest.raises(ValueError, match="Nobel"):
        llm.validate_narrative(bad, 2, STORIES)
    flipped = _text("La Borsa di Milano chiude in rialzo [1].")
    with pytest.raises(ValueError, match="aumento"):
        llm.validate_narrative(flipped, 2, STORIES)


def test_summarize_news_drops_ungrounded_summaries(monkeypatch):
    articles = [{"title": "Borsa: Milano chiude in calo", "summary": "Banche in rosso"},
                {"title": "Attore aggredito a Roma", "summary": "Ricoverato in terapia intensiva"}]
    reply = {"sintesi": [{"id": 0, "testo": "La Borsa di Milano chiude in rialzo."},
                         {"id": 1, "testo": "L'attore aggredito a Roma è in coma."}]}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": json.dumps(reply)}}

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: R())
    assert llm.summarize_news(articles) == {}                        # entrambe scartate: si userà il titolo del feed
    reply["sintesi"][0]["testo"] = "La Borsa di Milano chiude in calo con le banche."
    assert llm.summarize_news(articles) == {0: "La Borsa di Milano chiude in calo con le banche."}
