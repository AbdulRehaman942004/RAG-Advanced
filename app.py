import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import generation

def init_session_state() -> None:
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []


def main() -> None:
    st.set_page_config(
        page_title="Advanced RAG – PDF Q&A",
        page_icon="📘",
        layout="wide",
    )

    init_session_state()

    st.title("Advanced RAG – PDF Question Answering")
    st.caption(
        "Ask questions answered from a pre-indexed PDF collection in ChromaDB using the Groq LLM."
    )

    with st.sidebar:
        st.header("Configuration")

        n_results = st.slider(
            "Number of chunks to retrieve",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
            help="Chunks sent as context to the LLM per question.",
        )

        model_name = st.text_input(
            "Groq model name",
            value="openai/gpt-oss-20b",
            help="Model ID used via Groq's OpenAI-compatible API.",
        )

        st.markdown("---")
        st.subheader("Environment status")
        groq_key_present = bool(os.environ.get("GROQ_API_KEY"))
        st.write(f"**GROQ_API_KEY set:** {'✅' if groq_key_present else '❌'}")

    tab_overview, tab_qa = st.tabs(["Overview", "Ask Questions"])

    with tab_overview:
        st.subheader("How this app works")
        st.markdown(
            """
1. **Data**: This app is built specifically for the `Oxford-Guide-2022.pdf` document, which has been pre-indexed into ChromaDB under the collection name `Oxford-Guide-2022`.
2. **Ask questions** (tab *Ask Questions*): Type a question. The app retrieves relevant chunks from the `Oxford-Guide-2022` collection and answers using the Groq LLM with that context.
"""
        )
        st.success("**Current document:** `Oxford-Guide-2022.pdf` (collection `Oxford-Guide-2022`)")

    with tab_qa:
        st.subheader("Ask questions")

        st.info("Questions will be answered using only the `Oxford-Guide-2022` collection in ChromaDB.")

        question = st.text_input("Your question", placeholder="e.g. How old is Oxford University?")

        if st.button("Get Answer", type="primary") and question.strip():
            user_query = question.strip()

            # First, compute confidence score to decide whether to run RAG
            with st.spinner("Evaluating question relevance to Oxford..."):
                score = generation.confidence_score(
                    user_query=user_query,
                    model_name=model_name,
                )

            if score < 0.8:
                answer = "The question is irrelevant to Oxford Please come back with a relevant question."
                context_chunks = []
            else:
                with st.spinner("Retrieving context and generating answer..."):
                    answer, context_chunks = generation.run_rag_query(
                        user_query=user_query,
                        n_results=n_results,
                        model_name=model_name,
                    )

            st.markdown("#### Answer")
            st.write(answer)

            st.session_state.qa_history.insert(
                0,
                {
                    "question": user_query,
                    "answer": answer,
                    "collection": "Oxford-Guide-2022",
                },
            )

            if context_chunks:
                with st.expander("Show retrieved context chunks"):
                    for i, chunk in enumerate(context_chunks, start=1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(chunk)
                        st.markdown("---")

        if st.session_state.qa_history:
            st.markdown("#### Recent questions")
            for item in st.session_state.qa_history[:5]:
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                st.caption(f"Collection: `{item['collection']}`")
                st.markdown("---")


if __name__ == "__main__":
    main()
