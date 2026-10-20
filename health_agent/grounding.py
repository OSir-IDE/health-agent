"""Controlli deterministici: una frase generata dall'LLM non deve affermare più di quanto dicono le fonti.

Tre controlli, dal più sicuro al più rumoroso:
  1. termini delicati (morti, coma, Nobel, arresti, condanne…): se la frase li usa, almeno una fonte deve
     contenere un termine equivalente (italiano o inglese)          -> BLOCCA
  2. direzione (rialzo/ribasso, aumento/calo…): la frase non può dire l'opposto di quanto dicono le fonti
     (se le fonti contengono entrambe le direzioni non si decide)   -> BLOCCA
  3. nomi propri mai visti nelle fonti                              -> solo SEGNALATI nel registro

Limiti noti (da non dimenticare): non vede le inversioni di ruolo ("aggredito DA / SU"), né i dettagli che
non appartengono a nessuna delle liste. È una rete di sicurezza economica, non un verificatore di fatti.
"""
import re
import unicodedata

# (nome, radici che la frase generata usa, radici che almeno una fonte deve contenere: italiano e inglese)
CONCEPTS = [
    ("coma", ("coma",), ("coma",)),
    ("premio Nobel", ("nobel",), ("nobel",)),
    ("morti", ("mort", "uccis", "decess", "decedut", "strage", "trucid"),
     ("mort", "uccis", "decess", "decedut", "strage", "trucid", "vita", "kill", "dead", "death", "die",
      "murder", "massacre", "fatal", "perish", "lives")),
    ("feriti", ("ferit",), ("ferit", "injur", "wound", "hurt", "ricover", "hospital")),
    ("arresto", ("arrest", "fermat", "manett"),
     ("arrest", "fermat", "manett", "detain", "custod", "jail", "carcer", "cella", "prigion")),
    ("condanna", ("condann", "ergastol"),
     ("condann", "ergastol", "sentenc", "convict", "jail", "prison", "guilty", "carcer", "pena")),
    ("assoluzione", ("assolt", "assoluz"), ("assolt", "assoluz", "acquit", "cleared", "not guilty")),
    ("dimissioni", ("dimission", "dimess"), ("dimission", "dimess", "resign", "quit", "step down", "stepped down")),
    ("licenziamento", ("licenzi",),
     ("licenzi", "fired", "sacked", "dismiss", "laid off", "layoff", "let go", "esoner", "allontan")),
    ("attentato", ("attentat", "terroris"), ("attentat", "terror", "attack", "bomb", "esplos", "blast")),
    ("guerra", ("guerra",), ("guerra", "war", "conflit", "invas", "raid", "attacc", "attack", "bombard", "missil")),
    ("terapia intensiva", ("terapia intensiva",), ("terapia intensiva", "intensive care", "rianimaz", "critical")),
    ("sanzioni", ("sanzion",), ("sanzion", "sanction", "dazi", "tariff")),
]

UP = re.compile(r"\b(rialz\w*|aument\w*|cresc\w*|balz\w*|sal[ei]\w*|salgono|rise[sn]?|rose|gain\w*|jump\w*|"
                r"surg\w*|climb\w*|higher|up)\b")
DOWN = re.compile(r"\b(ribass\w*|calo|cal[ai]\w*|scend\w*|diminu\w*|crol\w*|cad[ei]\w*|perd\w*|fall\w*|fell|"
                  r"drop\w*|slump\w*|lower|down|decline\w*|lost|lose)\b")

# parole maiuscole che non sono nomi: istituzioni, luoghi generici, ecc. (per il solo registro dei nomi)
COMMON_CAPS = {"stati", "uniti", "unione", "europea", "italia", "paesi", "corte", "borsa", "serie", "governo",
               "ministro", "presidente", "premier", "camera", "senato", "parlamento", "papa", "chiesa",
               "tribunale", "procura", "polizia", "esercito", "europa", "mondo", "intanto", "sul", "fuori",
               "dopo", "mentre", "secondo", "anche", "sempre"}
WORD = re.compile(r"[A-Za-zÀ-ÿ0-9'’-]+")
CAP_WORD = re.compile(r"\b[A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]{3,}\b")


def norm(text: str) -> str:
    """minuscolo e senza accenti."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _has(text: str, stems) -> bool:
    return any(re.search(r"\b" + re.escape(s), text) for s in stems)


def ungrounded(sentence: str, source: str) -> list[str]:
    """Problemi che bloccano: termini delicati senza riscontro e direzione opposta alle fonti."""
    s, src = norm(sentence), norm(source)
    problems = [f"«{name}» non compare nelle fonti" for name, trigger, ok in CONCEPTS
                if _has(s, trigger) and not _has(src, ok)]
    s_up, s_down, src_up, src_down = UP.search(s), DOWN.search(s), UP.search(src), DOWN.search(src)
    if s_up and not s_down and src_down and not src_up:
        problems.append("la frase dice un aumento, le fonti un calo")
    if s_down and not s_up and src_up and not src_down:
        problems.append("la frase dice un calo, le fonti un aumento")
    return problems


def unseen_names(sentence: str, source: str) -> list[str]:
    """Nomi propri della frase (non a inizio frase) che nelle fonti non compaiono: solo per il registro."""
    src_stems = {norm(w)[:5] for w in WORD.findall(source)}
    words = [m for m in CAP_WORD.finditer(sentence)]
    out = []
    for m in words:
        before = sentence[:m.start()].rstrip()
        if not before or before[-1] in ".!?" or before.endswith("]"):
            continue                                   # maiuscola per inizio frase
        w = m.group()
        if norm(w) in COMMON_CAPS or norm(w)[:5] in src_stems:
            continue
        out.append(w)
    return list(dict.fromkeys(out))
