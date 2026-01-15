import streamlit as st
import uuid

from rag import answer_question, log_feedback
from feedback_store import store_feedback_example

POSITIVE_REASONS = [
    "Correct and accurate",
    "Well grounded in sources",
    "Clear and easy to understand",
    "Correct reasoning steps",
    "Appropriate level of detail",
]

NEGATIVE_REASONS = [
    "Incorrect or misleading answer",
    "Hallucinated information",
    "Missing relevant context",
    "Misunderstood the question",
    "Too vague or generic",
    "Too verbose or unfocused",
]


def main():
    st.title("SQL RAG APPLICATION")
    st.write("Ask any questions related to SQL")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"

    if "last_trace_id" not in st.session_state:
        st.session_state.last_trace_id = None

    if "last_question" not in st.session_state:
        st.session_state.last_question = None

    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None

    if "last_context" not in st.session_state:
        st.session_state.last_context = None

    if "show_feedback_form" not in st.session_state:
        st.session_state.show_feedback_form = False

    if "feedback_score" not in st.session_state:
        st.session_state.feedback_score = None

    question = st.text_input("Enter your question")

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            answer, sources, trace_id, context = answer_question(
                question=question.strip(),
                k=4,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
            )

        st.session_state.last_trace_id = trace_id
        st.session_state.last_question = question
        st.session_state.last_answer = answer
        st.session_state.last_context = context
        st.session_state.show_feedback_form = False
        st.session_state.feedback_score = None

        store_feedback_example(
            trace_id=trace_id,
            question=question.strip(),
            context=context,
            model_answer=answer,
            score=None,  
            reason=None,  
            comment=None,
        )

        st.subheader("Answer")
        st.write(answer)

        is_unknown = ("I do not know the answer" in answer or "No relevant context found" in answer or "I can’t generate jokes or poems" in answer)

        if not is_unknown and sources:
            st.subheader("Sources")

            source_pages = {}
            for src, page in sources:
                source_pages.setdefault(src, set()).add(page)

            for src, pages in source_pages.items():
                pages_text = ", ".join(str(p) for p in sorted(pages))
                st.write(f"{src}: {pages_text}")

    if st.session_state.last_trace_id:
        st.subheader("Was this answer helpful?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Yes", key="thumbs_up"):
                st.session_state.feedback_score = 1
                st.session_state.show_feedback_form = True

        with col2:
            if st.button("👎 No", key="thumbs_down"):
                st.session_state.feedback_score = 0
                st.session_state.show_feedback_form = True

    if st.session_state.show_feedback_form and st.session_state.feedback_score is not None:
        score = st.session_state.feedback_score

        if score == 1:
            st.markdown("### What did the assistant do well?")
            reason = st.selectbox(
                "Select one reason",
                POSITIVE_REASONS,
                key="positive_reason",
            )
        else:
            st.markdown("### What went wrong?")
            reason = st.selectbox(
                "Select one reason",
                NEGATIVE_REASONS,
                key="negative_reason",
            )

        if st.button("Submit feedback", key="submit_feedback"):

            log_feedback(st.session_state.last_trace_id, score)

            store_feedback_example(
                trace_id=st.session_state.last_trace_id,
                question=st.session_state.last_question,
                context=st.session_state.last_context,
                model_answer=st.session_state.last_answer,
                score=score,
                reason=reason,
                comment=None,
            )

            st.success("Thanks! Your feedback has been recorded.")

            st.session_state.show_feedback_form = False
            st.session_state.feedback_score = None


if __name__ == "__main__":
    main()
