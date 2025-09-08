import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import json
from datetime import datetime, timezone
from src.api_main import ChatMessage, UserMessage, AssistantMessage

# Configure the API endpoint
API_BASE_URL = "http://127.0.0.1:8000"  # Assuming FastAPI is running locally

st.title("ASK Chat")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("What is your question?"):
    # Create user message using Pydantic model
    user_msg = UserMessage(
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        content=prompt,
    )
    # Add user message to history
    st.session_state.messages.append(user_msg.model_dump())
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send request to API and stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # POST to /chat/ endpoint with form data
            response = requests.post(
                f"{API_BASE_URL}/chat/",
                data={"prompt": prompt},
                stream=True
            )
            response.raise_for_status()

            # Process NDJSON stream
            for line in response.iter_lines():
                if line:
                    try:
                        message_data = json.loads(line.decode('utf-8'))
                        if message_data.role == "assistant":
                            # Create assistant message using Pydantic model
                            assistant_msg = AssistantMessage(**message_data)
                            full_response = assistant_msg.content
                            message_placeholder.markdown(full_response)
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with API: {e}")

    # Add assistant response to history
    if full_response:
        assistant_msg = AssistantMessage(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            content=full_response,
        )
        st.session_state.messages.append(assistant_msg.model_dump())