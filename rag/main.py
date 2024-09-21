import streamlit as st
from pipelines.IndexerPipeline import IndexerPipeline
from pipelines.RAGPipeline import RAGPipeline
import time
import threading

# Streamed response emulator
def streaming_callback(chunk, container):
    if "response_text" not in st.session_state:
        st.session_state.response_text = ""
    st.session_state.response_text += chunk.content
    container.markdown(st.session_state.response_text)

def clear_chat():
    st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def main():
    st.title("Simple chat")
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
            indexer.run("./scraper/data/")

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
    if prompt := st.chat_input("What is up?"):
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