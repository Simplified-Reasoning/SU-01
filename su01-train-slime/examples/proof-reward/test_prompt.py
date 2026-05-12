import openai

user_prompt = """
# System Role: Deterministic Mathematical Autograder

You are a precise, automated grading system. Your sole function is to determine if the final answer provided in the `Model Solution` is mathematically equivalent to the `Golden Answer`. You must NOT grade the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)

Equivalence is mandatory for a correct grade. You must rigorously verify if the answers represent the exact same mathematical value or expression, even if the format differs.
- **Algebraic Equivalence:** e.g., `n(n+1)/2` is equivalent to `n^2/2 + n/2`. You must verify the algebra.
- **Numerical Equivalence:** e.g., `1/2` is equivalent to `0.5`; `sqrt(2)/2` is equivalent to `1/sqrt(2)`.
- **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order of elements does not matter (e.g., {{1, 2}} is equivalent to {{2, 1}}).
- **Partial Credit:** No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.
- **No Answers:** If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.

# 3. Output Protocol (Strict Compliance Required)

You must execute the task using a two-part structure. Failure to follow this structure will result in task failure.

**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags. Make your thinking concise. This section details your reasoning process and must follow these steps sequentially:
1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer based on the Extraction Protocol. If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. Detail the steps taken to verify mathematical equivalence (e.g., simplification, algebraic manipulation). You must actively try to prove they are the same before concluding they are different.
4. **Conclusion:** State the final determination ("Correct" or "Incorrect").

**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output **ONLY** the final grade.
- If Correct: \\boxed{{Correct}}
- If Incorrect: \\boxed{{Incorrect}}

**CRITICAL CONSTRAINT: Do not add any text, explanations, or formatting outside the <thinking> tags or the final \\boxed{{}} output.**

---

# 4. Input Data
Here is the problem, model solution, and golden answer to grade:

Problem: {Problem_Statement}
Extracted Model Answer: {Model_Answer}
Golden Answer: {Golden_Answer}
"""

# base url is http://10.102.247.45:34882/v1
client = openai.OpenAI(api_key="none", base_url="http://10.102.247.21:35999/v1")

user_prompt = user_prompt.format(Problem_Statement="试问当 $B_0$ 为何值时，总电流 $I_J = 0$？", Model_Answer="B_{0} = \\\\dfrac{n \\\\pi \\\\hbar}{e (d + 2\\\\lambda) Y} \\\\quad (n = \\\\pm 1, \\\\pm 2, \\\\dots)", Golden_Answer="$B_0 = n \\\\frac{\\\\pi \\\\hbar}{e (d + 2\\\\lambda)Y}$ where $n = \\\\pm 1, \\\\pm 2, ...$")

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "system", "content": "Reasoning: Low"},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.6,
    max_tokens=4096,
)

print(response.choices[0].message.content)