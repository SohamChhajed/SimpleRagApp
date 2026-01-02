import streamlit as st
from rag import answer_question
def main():
    st.title("SQL RAG APPLICATION")
    st.write("Ask any questions related sql")
    question=st.text_input("Enter your question")
    if st.button("Ask") and question.strip():
        answer,sources = answer_question(question.strip(), k=4)

        st.subheader("Answer")
        st.write(answer)

        is_unknown = ("I do not know the answer" in answer) or ("No relevant context found" in answer) or ("I can’t generate jokes or poems" in answer)

        if not is_unknown and sources:
            st.subheader("Sources")
            source_pages = {}
            for src, page in sources:
                source_pages.setdefault(src, set()).add(page)

            for src, pages in source_pages.items():
                pages_text = ", ".join(str(p) for p in sorted(pages))
                st.write(f"{src}: {pages_text}")

if __name__=="__main__":
    main()
