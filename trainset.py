import dspy

trainset = [
    dspy.Example(
        context=[
            "The inner join selects all rows from both tables as long as there is a match between the columns.",
            "Example: SELECT * FROM Emp INNER JOIN Dept ON Emp.ID = Dept.ID;"
        ],
        question="How does an Inner Join work and what is the syntax to join an Emp and Dept table on their ID?",
        answer="An Inner Join returns records that have matching values in both tables. The syntax is: SELECT * FROM Emp INNER JOIN Dept ON Emp.ID = Dept.ID;"
    ).with_inputs("context", "question"),
    dspy.Example(
        context=[
            "The RANK() function returns the position of any row in the specified partition.",
            "Syntax: SELECT RANK() OVER (PARTITION BY DEPTNAME ORDER BY SALARY DESC) AS DEPT_RANK FROM workers;"
        ],
        question="Write a SQL query to rank employee salaries within their respective departments.",
        answer="To rank salaries within departments, use: SELECT RANK() OVER (PARTITION BY DEPTNAME ORDER BY SALARY DESC) AS DEPT_RANK, ENAME, SALARY FROM workers;"
    ).with_inputs("context", "question"),
    dspy.Example(
        context=[
            "The REPLACE() string function replaces all occurrences of a source substring with a target substring.",
            "Input: SELECT REPLACE('WELCOME TO DATA SCIENCE', 'DATA', 'INFORMATION') returns 'WELCOME TO INFORMATION SCIENCE'."
        ],
        question="How do you change the word 'DATA' to 'INFORMATION' in a string using SQL?",
        answer="You use the REPLACE() function. Example: SELECT REPLACE('WELCOME TO DATA SCIENCE', 'DATA', 'INFORMATION');"
    ).with_inputs("context", "question"),
    dspy.Example(
        context=[
            "The COALESCE function replaces NULL values with a default value.",
            "The first parameter is the column that may be NULL, the second is the replacement value."
        ],
        question="If a 'quantity' column has NULL values, how can I replace them with -1 in my results?",
        answer="You can use the COALESCE function: SELECT COALESCE(quantity, -1) AS quantity FROM sales;"
    ).with_inputs("context", "question"),
    dspy.Example(
        context=[
            "The mode is the value that appears most frequently. SQL does not have a built-in mode function.",
            "To calculate it: SELECT TOP 1 ColumnName FROM TableName GROUP BY ColumnName ORDER BY COUNT(*) DESC."
        ],
        question="SQL doesn't have a MODE() function. How can I find the most frequent value in a column?",
        answer="You can find the mode by grouping the column, counting the occurrences, and ordering by that count in descending order. Use: SELECT ColumnName FROM TableName GROUP BY ColumnName ORDER BY COUNT(*) DESC LIMIT 1;"
    ).with_inputs("context", "question")
]

