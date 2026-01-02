import streamlit as st
from rag import answer_question
def main():
    st.title("SQL RAG APPLICATION")
    st.write("Ask any questions related sql")
    question=st.text_input("Enter your question")
    if st.button("Ask") and question.strip():
        answer,sources=answer_question(question.strip(),k=4)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Sources")
        if sources:
            source_pages = {}

            for src, page in sources:
                if src not in source_pages:
                    source_pages[src] = set()
                source_pages[src].add(page)

            for src, pages in source_pages.items():
                pages_sorted = sorted(pages)
                pages_text = ", ".join(str(p) for p in pages_sorted)
                st.write(f"{src}: {pages_text}")
if __name__=="__main__":
    main()
