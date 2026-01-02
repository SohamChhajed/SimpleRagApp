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
            for src,page in sources:
                st.write(f"-{src}(page{page})")
if __name__=="__main__":
    main()
