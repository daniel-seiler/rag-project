import streamlit as st
from rag.pipelines.IndexerPipeline import IndexerPipeline
from rag.pipelines.RAGPipeline import RAGPipeline
import time
import threading
from streamlit.delta_generator import DeltaGenerator

from typing import Iterator
from haystack.dataclasses import Document, ChatMessage

# Streamed response emulator
def streaming_callback(chunk:ChatMessage, container:DeltaGenerator) -> None:
    """
    Callback function to handle streaming chat messages and update the response text in the session state.

    Args:
        chunk (ChatMessage): A chunk of the chat message containing the content to be appended.
        container (DeltaGenerator): A Streamlit DeltaGenerator object used to update the UI with the new response text.

    Returns:
        None
    """
    if "response_text" not in st.session_state:
        st.session_state.response_text = ""
    st.session_state.response_text += chunk.content
    container.markdown(st.session_state.response_text)

def clear_chat() -> None:
    """
    Clears the chat messages stored in the Streamlit session state.

    This function resets the `messages` list in the `st.session_state` to an empty list.
    It then iterates over the cleared `messages` list (which will be empty) and attempts
    to display each message using Streamlit's chat message functionality.

    Returns:
        None
    """
    st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def response_generator(response: str) -> Iterator[str]:
    """
    Generates a sequence of words from the given response string, yielding each word followed by a space.

    Args:
        response (str): The input string to be split into words.

    Yields:
        Iterator[str]: An iterator that yields each word followed by a space, with a delay of 0.05 seconds between each word.
    """
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def main():
    st.title("PCL-Chatbot")
    if st.button("Clear chat"):
        clear_chat()
    indexer = IndexerPipeline()
    generator = RAGPipeline()
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if st.button("Index documents"):
        progress_text = "Running preprocessing pipeline and indexing documents..."
        my_bar = st.progress(0, text=progress_text)
        def run_indexer():
            indexer.run("data/")

        indexer_thread = threading.Thread(target=run_indexer)
        indexer_thread.start()
        while not indexer.pipeline_status_done:
            time.sleep(0.1)  # Sleep for a short duration to avoid busy waiting
            my_bar.progress(indexer.get_progress(), text=progress_text)
        my_bar.progress(100, text="Indexing complete!")


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    if prompt := st.chat_input("Ask me questions about PCL!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        response_container = st.chat_message("assistant").empty()
        st.session_state.response_text = ""
        generator.llm_generator.streaming_callback = lambda chunk: streaming_callback(chunk, response_container)

        success, output = generator.run(prompt, st.session_state.messages)
        if not success:
            response = st.write_stream(response_generator(output))
            st.session_state.messages.append({"role": "system", "content": response})

        else:
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.response_text})
        


if __name__ == "__main__":
    main()