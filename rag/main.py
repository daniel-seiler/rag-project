import streamlit as st
from pipelines.IndexerPipeline import IndexerPipeline
from pipelines.RAGPipeline import RAGPipeline

# Streamed response emulator
def streaming_callback(chunk, container):
    if "response_text" not in st.session_state:
        st.session_state.response_text = ""
    st.session_state.response_text += chunk.content
    container.markdown(st.session_state.response_text)


def main():
    st.title("Simple chat")
    indexer = IndexerPipeline()
    generator = RAGPipeline()
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline_initialized" not in st.session_state:
        st.session_state["pipeline_initialized"] = False
    if not st.session_state["pipeline_initialized"]:
        indexer.run("./data/genai/")
        st.session_state["pipeline_initialized"] = True
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

        generator.run(prompt, st.session_state.messages)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.response_text})


if __name__ == "__main__":
    main()