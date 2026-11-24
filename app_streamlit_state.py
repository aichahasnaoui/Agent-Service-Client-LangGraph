import uuid
from pathlib import Path

import streamlit as st

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent_boutique_state import app
from outils_base import creer_base


# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Boutique | Service Client",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. DESIGN CSS
# ==================================================

st.markdown("""
<style>

/* Fond principal */
.stApp {
    background: #101827;
    color: #F1F5F9;
}

/* Barre latérale */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172337, #101827);
    border-right: 1px solid #29384D;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF;
}

/* Titres */
h1, h2, h3 {
    color: #FFFFFF !important;
}

/* Texte */
p, label {
    color: #CBD5E1;
}

/* Boutons */
.stButton > button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
    transform: translateY(-2px);
}

/* Cartes */
.info-card {
    background: linear-gradient(135deg, #1E293B, #25344B);
    border: 1px solid #34445D;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

/* Bannière principale */
.hero {
    background: linear-gradient(120deg, #4338CA, #7C3AED, #9333EA);
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
}

.hero h1 {
    color: white !important;
    font-size: 32px;
    margin-bottom: 8px;
}

.hero p {
    color: #E0E7FF;
    font-size: 15px;
}

/* Métriques */
div[data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 12px;
}

/* Séparateurs */
hr {
    border-color: #334155;
}

/* Questions suggérées */
.question-box {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 9px;
    color: #E2E8F0;
    font-size: 13px;
}

/* Zone de saisie */
div[data-testid="stChatInput"] {
    border: 1px solid #475569;
    border-radius: 14px;
}

/* Messages */
div[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 12px;
}

/* Alertes */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Masquer le menu supérieur */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. INITIALISATION DE LA BASE
# ==================================================

@st.cache_resource
def initialiser_base():
    return creer_base()


initialiser_base()


# ==================================================
# 4. INITIALISATION DE LA CONVERSATION
# ==================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ==================================================
# 5. RECUPERATION DE L'ETAT LANGGRAPH
# ==================================================

snapshot = app.get_state(config)

etat = snapshot.values

actions = (
    snapshot.interrupts[0].value
    if snapshot.interrupts
    else None
)


# ==================================================
# 6. FONCTION POUR EXTRAIRE LE TEXTE
# ==================================================

def extraire_texte(message):

    contenu = message.content

    if isinstance(contenu, str):
        return contenu

    if isinstance(contenu, list):

        textes = []

        for bloc in contenu:

            if isinstance(bloc, dict):

                if bloc.get("type") == "text":
                    textes.append(bloc.get("text", ""))

        return "\n".join(textes)

    return str(contenu)


# ==================================================
# 7. BARRE LATERALE
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 0;">
            <div style="font-size:45px;">🛍️</div>
            <h2 style="margin-bottom:0;">Boutique</h2>
            <p style="color:#94A3B8; font-size:13px;">
                Intelligent Customer Service
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Nouvelle conversation
    if st.button(
        "＋ Nouvelle conversation",
        use_container_width=True
    ):

        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()

    st.markdown("### 🤖 Agent intelligent")

    st.caption(
        "Assistant basé sur LangGraph, RAG, SQL et validation humaine."
    )

    # Graphe
    with st.expander("📊 Architecture de l'agent"):

        chemin_graphe = Path(__file__).parent / "graphe_agent.png"

        if chemin_graphe.exists():

            st.image(str(chemin_graphe), use_container_width=True)

        else:

            st.info(
                "Le fichier graphe_agent.png n'existe pas encore."
            )

    st.divider()

    # Etat du graphe
    st.markdown("### 📈 État du système")

    st.metric(
        "Appels au LLM",
        etat.get("nb_appels_llm", 0)
    )

    st.divider()

    # Journal des validations
    st.markdown("### 📝 Journal des validations")

    journal = etat.get("journal", [])

    if journal:

        for ligne in journal:
            st.markdown(
                f"<div class='question-box'>✅ {ligne}</div>",
                unsafe_allow_html=True
            )

    else:

        st.caption("Aucune action sensible.")

    st.divider()

    # Questions suggérées
    st.markdown("### 💡 Questions à essayer")

    questions = [
        "Quels sont les délais de livraison ?",
        "Où en est la commande 1002 ?",
        "Annule ma commande 1001.",
        "Puis-je retourner la commande 1003 ?",
        "Je veux retourner la commande 1004."
    ]

    for q in questions:

        st.markdown(
            f"<div class='question-box'>💬 {q}</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="text-align:center; padding-top:20px;">
            <p style="font-size:11px; color:#64748B;">
                Powered by LangGraph + Gemini
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 8. PAGE PRINCIPALE
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛒 Service Client Intelligent</h1>
        <p>
            Bienvenue dans notre boutique !
            Posez vos questions concernant vos commandes,
            les livraisons et les retours de produits.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 9. INFORMATIONS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>📚 Base documentaire</h3>
            <p>Recherche intelligente avec RAG et ChromaDB.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🗄️ Gestion SQL</h3>
            <p>Consultation et gestion des commandes avec SQLite.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <h3>👤 Validation humaine</h3>
            <p>Confirmation des actions sensibles par un responsable.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 10. AFFICHAGE DE LA CONVERSATION
# ==================================================

st.markdown("### 💬 Votre conversation")

if not etat.get("messages"):

    st.info(
        "Bonjour ! 👋 Je suis votre assistant virtuel. "
        "Comment puis-je vous aider aujourd'hui ?"
    )


for message in etat.get("messages", []):

    if message.type == "human":

        with st.chat_message("user", avatar="👤"):

            st.markdown(extraire_texte(message))

    elif message.type == "ai":

        texte = extraire_texte(message)

        if texte:

            with st.chat_message("assistant", avatar="🤖"):

                st.markdown(texte)


# ==================================================
# 11. VALIDATION HUMAINE
# ==================================================

envoi_au_graphe = None

if actions:

    st.divider()

    st.warning(
        "⚠️ Une action nécessite la validation d'un responsable."
    )

    st.markdown("### 🔐 Validation requise")

    for action in actions:

        st.code(action)

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "✅ Valider l'action",
            use_container_width=True
        ):

            envoi_au_graphe = Command(resume="oui")

    with col2:

        if st.button(
            "❌ Refuser l'action",
            use_container_width=True
        ):

            envoi_au_graphe = Command(resume="non")


# ==================================================
# 12. SAISIE DE LA QUESTION
# ==================================================

question = st.chat_input(
    "Écrivez votre question ici...",
    disabled=bool(actions)
)

if question:

    envoi_au_graphe = {
        "messages": [
            HumanMessage(content=question)
        ]
    }


# ==================================================
# 13. EXECUTION DU GRAPHE LANGGRAPH
# ==================================================

if envoi_au_graphe is not None:

    with st.spinner("🤖 Votre assistant prépare une réponse..."):

        try:

            app.invoke(
                envoi_au_graphe,
                config
            )

            st.rerun()

        except Exception as erreur:

            st.error(
                f"Une erreur est survenue : {erreur}"
            )


# ==================================================
# 14. PIED DE PAGE
# ==================================================

st.markdown(
    """
    <br>
    <hr>
    <div style="text-align:center; color:#64748B; font-size:8px;">
        🛍️ Boutique Intelligent Assistant |
        LangGraph • RAG • ChromaDB • SQLite • Gemini
    </div>
    """,
    unsafe_allow_html=True
)