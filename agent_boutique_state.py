import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from indexer import connecter_chroma
from outils_base import OUTILS_SQL


# ==================================================
# 1. CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ==================================================

load_dotenv()


# ==================================================
# 2. CONNEXION A CHROMADB
# ==================================================

vector_store = connecter_chroma()


# ==================================================
# 3. OUTIL DE RECHERCHE DANS LE PDF (RAG)
# ==================================================

@tool
def chercher_politique(question: str) -> str:
    """
    Cherche dans les règles de la boutique :
    livraison, retour, annulation et paiement.
    """

    docs = vector_store.similarity_search(question, k=2)

    if not docs:
        return "Aucune information trouvée dans la politique de la boutique."

    return "\n".join(d.page_content for d in docs)


# ==================================================
# 4. LISTE DES OUTILS
# ==================================================

OUTILS = [
    chercher_politique,
    *OUTILS_SQL
]


# Outils nécessitant une validation humaine
SENSIBLES = {
    "annuler_commande",
    "creer_retour"
}


# ==================================================
# 5. PROMPT SYSTEME
# ==================================================

PROMPT = (
    "Tu es l'assistant du service client d'une boutique en ligne tunisienne. "

    "Réponds en français, brièvement et poliment. "

    "N'invente jamais d'information. "

    "Pour une question sur les règles de la boutique, "
    "utilise obligatoirement chercher_politique. "

    "Pour consulter une commande, utilise consulter_commande. "

    "Avant toute annulation ou tout retour : "
    "consulte la commande, vérifie la règle de la boutique, "
    "puis agis seulement si la règle est respectée. "

    "Si la règle n'est pas respectée, explique pourquoi au client. "

    "Ne prétends jamais avoir effectué une action "
    "si l'outil ne l'a pas confirmée."
)


# ==================================================
# 6. INITIALISATION DU MODELE GEMINI
# ==================================================

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODELE"),
    temperature=0
)
# ==================================================
# 7. DEFINITION DE L'ETAT DU GRAPHE
# ==================================================

class Etat(TypedDict):

    # Les nouveaux messages sont ajoutés à l'historique
    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    # Les éléments du journal sont concaténés
    journal: Annotated[
        list[str],
        operator.add
    ]

    # Nombre d'appels au modèle
    nb_appels_llm: int


# ==================================================
# 8. NOEUD AGENT
# ==================================================

def agent(state: Etat):

    reponse = llm.invoke(
        [SystemMessage(content=PROMPT)] + state["messages"]
    )

    return {
        "messages": [reponse],
        "nb_appels_llm": state.get("nb_appels_llm", 0) + 1
    }


# ==================================================
# 9. ROUTAGE DES ACTIONS
# ==================================================

def router(state: Etat):

    appels = state["messages"][-1].tool_calls

    # Aucun outil demandé : fin de conversation
    if not appels:
        return END

    # Si un outil sensible est demandé
    # on passe par la validation humaine
    if any(a["name"] in SENSIBLES for a in appels):
        return "validation"

    # Sinon, exécution directe des outils
    return "outils"


# ==================================================
# 10. VALIDATION HUMAINE
# ==================================================

def validation(state: Etat):

    appels = state["messages"][-1].tool_calls

    actions = [
        f"{a['name']} {a['args']}"
        for a in appels
    ]

    # Interruption du graphe pour demander une validation
    decision = interrupt(actions)

    if isinstance(decision, str) and decision.strip().lower() in ("oui", "o"):

        return Command(
            goto="outils",
            update={
                "journal": [
                    f"✅ validé : {x}"
                    for x in actions
                ]
            }
        )

    # Si la demande est refusée
    refus = [
        ToolMessage(
            content="Action refusée par le responsable.",
            tool_call_id=a["id"],
            name=a["name"],
            status="error"
        )
        for a in appels
    ]

    return Command(
        goto="agent",
        update={
            "messages": refus,
            "journal": [
                f"❌ refusé : {x}"
                for x in actions
            ]
        }
    )


# ==================================================
# 11. CONSTRUCTION DU GRAPHE LANGGRAPH
# ==================================================

graphe = StateGraph(Etat)


# Ajout des noeuds
graphe.add_node("agent", agent)

graphe.add_node(
    "validation",
    validation,
    destinations=("outils", "agent")
)

graphe.add_node(
    "outils",
    ToolNode(OUTILS)
)


# Définition des connexions
graphe.add_edge(START, "agent")

graphe.add_conditional_edges(
    "agent",
    router,
    ["outils", "validation", END]
)

graphe.add_edge("outils", "agent")


# ==================================================
# 12. COMPILATION DU GRAPHE
# ==================================================

app = graphe.compile(
    checkpointer=InMemorySaver()
)


print("✅ Agent LangGraph créé avec succès !")