import dspy
from feedback_store import get_psycopg2_conn  # Replace with your actual import


def load_feedback_trainset(max_samples=4, positive_ratio=0.25):
    conn = get_psycopg2_conn()
    cur = conn.cursor()

    num_positive = int(max_samples * positive_ratio)
    num_non_positive = max_samples - num_positive

    trainset = []

    cur.execute("""
        SELECT question, context, model_answer, score, reason
        FROM rag_feedback_examples
        WHERE score = 1
        ORDER BY created_at DESC
        LIMIT %s
    """, (num_positive,))
    
    positives = cur.fetchall()
    
    for question, context, model_answer, score, reason in positives:
        trainset.append(
            dspy.Example(
                question=question,
                context=context.split("\n\n") if isinstance(context, str) else context,
                answer=model_answer,
                human_score=float(score),
                human_feedback=reason or "Good answer",
            ).with_inputs("question", "context")
        )
    cur.execute("""
        SELECT question, context, model_answer, score, reason
        FROM rag_feedback_examples
        WHERE score IS DISTINCT FROM 1
        ORDER BY created_at DESC
        LIMIT %s
    """, (num_non_positive,))
    
    non_positives = cur.fetchall()
    
    for question, context, model_answer, score, reason in non_positives:
        example_data = {
            "question": question,
            "context": context.split("\n\n") if isinstance(context, str) else context,
            "answer": model_answer,
        }
        if score is not None:
            example_data["human_score"] = float(score)
            example_data["human_feedback"] = reason or "Incorrect answer"
        
        trainset.append(
            dspy.Example(**example_data).with_inputs("question", "context")
        )
    cur.close()
    conn.close()

    return trainset

# import dspy
# from feedback_store import get_psycopg2_conn


# def load_feedback_trainset(max_samples=4,positive_ratio=0.5):

#     conn = get_psycopg2_conn()
#     cur = conn.cursor()

#     num_positive = int(max_samples * positive_ratio)
#     num_negative = max_samples - num_positive

#     # Good answers
#     cur.execute("""
#         SELECT question, context, model_answer, score, reason
#         FROM rag_feedback_examples
#         WHERE score = 1
#         ORDER BY created_at DESC
#         LIMIT %s
#     """, (num_positive,))

#     positives = cur.fetchall()

#     # Bad answers
#     cur.execute("""
#         SELECT question, context,model_answer, score, reason
#         FROM rag_feedback_examples
#         WHERE score = 0
#         ORDER BY created_at DESC
#         LIMIT %s
#     """, (num_negative,))

#     negatives = cur.fetchall()

#     cur.close()
#     conn.close()

#     trainset = []

#     for question, context, model_answer,score,reason in positives + negatives:
#         trainset.append(
#             dspy.Example(
#                 question=question,
#                 context=context.split("\n\n"),
#                 answer=model_answer,
#                 human_score=float(score),
#                 human_feedback=reason,
#             ).with_inputs("question", "context")
#         )

#     return trainset
