import asyncio
import time
import streamlit as st
from agents import *


def map_role(role):
    # Mapowanie ról z formatu Google na format Streamlit
    return "assistant" if role == "model" else role


def fetch_agent_response(user_query, runner, user_id, session_id):
    """
    Uruchamia asynchronicznego agenta wewnątrz synchronicznego Streamlita.
    """
    # Tworzymy nową pętlę zdarzeń dla tego wątku
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Wywołujemy Twoją funkcję call_agent, którą masz już w kodzie
    # Zakładam, że call_agent jest dostępna w tym samym pliku lub zaimportowana
    response_text = loop.run_until_complete(
        call_agent(user_query, runner, user_id, session_id)
    )
    return response_text


def response_generator(response):
    # Używamy split(" "), aby NIE usuwać znaków nowej linii (\n)
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05) # 0.05 może być nieco wolne przy długich opisach