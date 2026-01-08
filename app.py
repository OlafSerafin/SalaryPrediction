import streamlit as st
from functions import *
from agents import *

st.set_page_config(page_title="Salary Prediction Agent", page_icon="🤖")

@st.cache_resource
def startup():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(setup_system())


runner, USER_ID, SESSION_ID = startup()

st.title('Salary Prediction Agent')

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Enter the job description")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent is processing the data..."):
            agent_text = fetch_agent_response(user_input, runner, USER_ID, SESSION_ID)

        st.write_stream(response_generator(agent_text))

    st.session_state.messages.append({"role": "assistant", "content": agent_text})
    st.rerun()