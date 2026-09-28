import streamlit as st

from rag_pipeline import ask_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 RAG Document Assistant")

st.write(
    "Ask questions about the information available "
    "in the knowledge document."
)


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_input(
    "Enter your question:"
)


# --------------------------------------------------
# Ask button
# --------------------------------------------------

if st.button("Ask"):

    if question.strip() == "":
        
        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the document and generating an answer..."
        ):

            answer, documents = ask_question(
                question
            )


        # ------------------------------------------
        # Display answer
        # ------------------------------------------

        st.subheader("Answer")

        st.success(answer)


        # ------------------------------------------
        # Display retrieved documents
        # ------------------------------------------

        with st.expander(
            "View Retrieved Documents"
        ):

            for i, document in enumerate(
                documents,
                start=1
            ):

                st.markdown(
                    f"### Document {i}"
                )

                st.write(document)