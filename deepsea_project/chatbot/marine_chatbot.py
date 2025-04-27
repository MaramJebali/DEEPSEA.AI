from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langdetect import detect
from duckduckgo_search import DDGS
import os

# === Memory Configuration ===
conversation_memory = []
MAX_MEMORY = 3  # last 3 user-bot turns

MARINE_KEYWORDS = [
    "ocean", "marine", "fish", "coral", "reef", "plankton", "ecosystem", "biodiversity",
    "sea", "pollution", "climate", "acidification", "species", "coast", "seagrass",
    "plastic", "conservation", "mammals", "jellyfish", "tuna", "microplastics", "algae"
]

# === Gentle Input Categories ===
GREETINGS = [
    "hello", "hi", "hey", "bonjour", "salut", "hola", "yo", "heya", "hi there",
    "good morning", "good afternoon", "good evening"
]

THANKS = [
    "thanks", "thank you", "thank u", "ty", "merci", "thanks a lot", "many thanks",
    "thx", "much appreciated", "gracias", "danke", "by", "bye"
]

NEUTRAL_POLITE = [
    "ok", "okay", "good", "great", "awesome", "nice", "cool", "yes", "no",
    "alright", "yep", "yup", "sure", "fine", "bien", "super", "parfait", "d'accord", "c'est bon"
]

# === Load LLM only once globally ===
llm = Ollama(
    model="mistral",
    temperature=0.7,
    top_p=0.9,
    top_k=50
)
def is_marine_related(query):
    return any(keyword in query.lower() for keyword in MARINE_KEYWORDS)
# === Load Vectorstore ONLY when needed ===
def load_retriever():
    embeddings = OllamaEmbeddings(model="mistral")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "vectorstore")
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# === DuckDuckGo Fallback Search ===
def fallback_duckduckgo(query, lang="en"):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, region='fr-fr' if lang == 'fr' else 'wt-wt', safesearch='moderate', max_results=3))
        return "\n".join([r['body'] for r in results if 'body' in r])

# === Main Response Function ===
def get_response(user_query):
    global conversation_memory
    query = user_query.lower().strip()

    # Load retriever at each query (safe for Django)
    retriever = load_retriever()

    # Empty input handler
    if not query:
        return "💬 Please ask a question about marine ecosystems so I can help you (EN/FR)."

    if query in GREETINGS:
        lang = "fr" if query in ["bonjour", "salut"] else "en"
        return "👋 Bonjour ! Comment puis-je vous aider à propos des écosystèmes marins ?" if lang == "fr" else "👋 Hello! How can I help you with marine ecosystems today?"

    elif query in THANKS:
        lang = "fr" if "merci" in query else "en"
        return "😊 Avec plaisir ! Si vous avez d'autres questions, je suis là pour vous aider." if lang == "fr" else "😊 You're very welcome! I'm always here to help you."

    elif query in NEUTRAL_POLITE:
        lang = detect(query)
        return "✨ C’est noté ! Si vous avez une question sur les écosystèmes marins, n’hésitez pas." if lang == "fr" else "✨ Got it! Let me know if you’d like to know more about marine life."

    # Language detection
    lang = detect(user_query)
    if lang not in ["en", "fr"]:
        return "❌ Sorry, I can only understand and respond in English or French. Please rephrase your question." \
            if lang == "en" else "❌ Désolé, je ne peux comprendre que l'anglais et le français. Veuillez reformuler votre question."

    # Retrieve knowledge
    context = retriever.invoke(user_query)
    context_text = "\n".join([doc.page_content for doc in context])
    fallback_used = False

    if not context_text.strip() and is_marine_related(query):
        fallback_used = True
        context_text = fallback_duckduckgo(user_query, lang)

    online_note = {
        "en": "🔎 I looked online to complete my answer as this wasn’t in my knowledge base.",
        "fr": "🔎 J'ai complété ma réponse grâce à des sources en ligne, car cela n'était pas présent dans ma base de connaissances."
    }[lang] if fallback_used else ""

    # Manage memory
    conversation_memory.append(f"User: {user_query}")
    if len(conversation_memory) > MAX_MEMORY * 2:
        conversation_memory = conversation_memory[-MAX_MEMORY * 2:]
    memory_text = "\n".join(conversation_memory)

    # === Prompt Construction ===
    if lang == "en":
        prompt = f"""
You are "Maren", a highly trained marine ecosystems expert AI. You communicate clearly, concisely, and kindly like a friendly biologist would. Stay focused solely on marine-related questions: oceans, coral reefs, marine species, pollution, biodiversity, climate impacts, ecosystems, etc.

Refuse off-topic questions gently and encourage rephrasing to stay within marine themes.

Your behavior guide:
- If the user greets you, respond warmly but briefly.
- If the user thanks you, acknowledge it with warmth.
- If information was retrieved via online search, mention that transparently but humbly.
- Use retrieved documents and recent conversation history to ground your answer.
- Never invent facts. If uncertain, say so clearly and helpfully.
- Maximum response length: **60 words**
- Answer in the same language as the user’s input (English or French).

{online_note}

---
🧠 Conversation Memory:
{memory_text}
---
📚 Retrieved Knowledge:
{context_text}
---
❓User Question:
{user_query}
---
Your Response:
"""
    else:
        prompt = f"""
Vous êtes "Maren", une intelligence artificielle experte en écosystèmes marins. Vous communiquez avec clarté, concision et bienveillance, comme un biologiste marin amical. Répondez uniquement aux questions liées à l’océan, aux récifs coralliens, à la biodiversité marine, à la pollution, au climat et aux écosystèmes.

Refusez poliment les questions hors sujet en invitant l’utilisateur à revenir à des sujets marins.

Guide comportemental :
- Si l'utilisateur vous salue, répondez brièvement et chaleureusement.
- S’il vous remercie, répondez avec bienveillance.
- Si l’information provient d’une recherche en ligne, mentionnez-le avec transparence.
- Appuyez vos réponses sur les documents retrouvés et la mémoire de conversation.
- Ne jamais inventer de faits. Si vous n’êtes pas sûr, dites-le.
- Longueur maximale : **60 mots**
- Répondez dans la langue de la question.

{online_note}

---
🧠 Mémoire de conversation :
{memory_text}
---
📚 Connaissances retrouvées :
{context_text}
---
❓Question utilisateur :
{user_query}
---
Votre réponse :
"""

  
    # Get LLM response
    response = llm.invoke(prompt).strip()

    # Update memory
    conversation_memory.append(f"Bot: {response}")
    if len(conversation_memory) > MAX_MEMORY * 2:
        conversation_memory = conversation_memory[-MAX_MEMORY * 2:]

    return response