import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Valencia Local Planner", page_icon="🦇")
st.title("🦇 Valencia Local Planner")

# --- RESUMEN DE CAPACIDADES (Debajo del título) ---
st.info("Bienvenido a tu asistente inteligente para Valencia.")

with st.expander("🔍 ¿Qué puede hacer este agente?"):
    st.markdown("""
    Este agente combina la potencia de **Gemini 2.5 Flash** con acceso a información externa para ayudarte en tu día a día:
    
    * **Búsqueda en Tiempo Real**: Localiza eventos, horarios de monumentos y noticias actuales en Valencia mediante **DuckDuckGo**.
    * **Conocimiento Enciclopédico**: Proporciona contexto histórico y técnico consultando **Wikipedia**.
    * **Memoria de Conversación**: Recuerda tus preguntas anteriores para mantener un hilo lógico.
    * **Asistente Local y Técnico**: Optimizado para buscar planes familiares, eventos de networking en la ciudad o conceptos del Máster en IA.
    """)

# --- FUNCIÓN DE LIMPIEZA DE SALIDA ---
def ensure_string_output(agent_result: dict) -> dict:
    """Extrae el texto limpio de la respuesta compleja del agente."""
    output_value = agent_result.get('output')
    if isinstance(output_value, list):
        concatenated_text = ""
        for item in output_value:
            if isinstance(item, dict) and item.get('type') == 'text':
                concatenated_text += item.get('text', '')
            elif isinstance(item, str):
                concatenated_text += item
        agent_result['output'] = concatenated_text
    elif not isinstance(output_value, str):
        agent_result['output'] = str(output_value)
    return agent_result

# --- SIDEBAR: GESTIÓN DE API KEY ---
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Google API Key", type="password")
    if st.button("🚀 Inicializar Agente", use_container_width=True):
        if api_key:
            st.session_state.agent_ready = True
            st.success("Agente listo")
        else:
            st.error("Introduce la API Key")

# --- LÓGICA PRINCIPAL DEL CHAT ---
if st.session_state.agent_ready:
    # Definición de herramientas
    search = DuckDuckGoSearchResults()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [search, wikipedia]
    
    # Modelo y Prompt
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente experto en Valencia. Responde de forma amigable y directa."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Creación del agente
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_cleaned = agent_executor | RunnableLambda(ensure_string_output)

    # Memoria persistente en Streamlit
    msgs = StreamlitChatMessageHistory(key="chat_history")
    agent_with_history = RunnableWithMessageHistory(
        agent_cleaned,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Mostrar historial de mensajes
    for msg in msgs.messages:
        role = msg.type if hasattr(msg, "type") else "assistant"
        st.chat_message(role).write(msg.content)

    # Input del usuario
    if user_query := st.chat_input("¿Qué planes hay para hoy?"):
        st.chat_message("human").write(user_query)
        with st.chat_message("assistant"):
            response = agent_with_history.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": "valencia_sess"}}
            )
            st.write(response["output"])
else:
    st.divider()
    st.warning("👈 Por favor, configura tu API Key en el menú lateral para activar el chat.")