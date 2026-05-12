SELF_REFINEMENT_PROMPT = """Your task is to refine a given solution to a problem. The problem may involve mathematics, physics, or another technical scientific domain. Your solution should be as accurate, rigorous, and easy-to-follow as possible.

Your final solution to the problem should be comprehensive and well-justified, which will be rated according to the following evaluation instruction:

''' txt
Here is the instruction to evaluate the quality of a solution to a problem. The problem may involve mathematics, physics, or another technical domain.

Please evaluate the solution according to the following criteria:
- If the solution is correct, well-justified, directly addresses the problem, and states the relevant assumptions or conditions when needed, then the score is 1
- If the solution is generally reasonable but contains minor errors, incomplete justification, omitted conditions, or insufficient explanation of key steps, then the score is 0.5
- If the solution fails to address the problem, contains serious mathematical/physical/factual errors, relies on invalid reasoning, or omits essential justification, then the score is 0

Additionally, merely stating a known formula, theorem, law, or standard result does not by itself remove the need for justification when such justification is necessary for the problem at hand. It is acceptable to use standard results, but only if they are applied correctly and under appropriate assumptions or conditions. If the solution ignores these assumptions or misapplies the result, it should be scored according to the criteria above, and definitely not with a score of 1.
'''

In fact, you are expected to reason carefully about the problem, examine the draft solution critically, and refine it by fixing issues you identify until you can make no further progress.

In your final response, you should present your best refined solution to the problem.
- To give a good final response, you should try your best to locate potential issues in the draft solution, including mathematical, physical, factual, or logical errors, and fix them as many as you can.
- A good final response should faithfully present the strongest solution you can produce after critical review of the draft.
- If the problem depends on assumptions, conventions, approximations, boundary conditions, domains of validity, or interpretation, you should state them clearly when relevant.
- Only when you fail to locate any meaningful issue in your solution should you treat it as fully correct.
- If you do notice unresolved issues, ambiguity, or uncertainty, it is totally ok to reflect that carefully in the final solution rather than pretending the solution is more certain than it really is.
- The worst final response would provide a flawed scientific solution but present it as unquestionably correct without careful checking. A better version should faithfully identify uncertainty, missing conditions, or limitations when they matter. Remember! You CAN’T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution // Your final solution should start with this exact same markdown title
... // Your best refined solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.

---

Here is your task input:

## Problem
{original_content}

## Candidate Solution to Refine
Here is a draft solution. You should provide a better solution by correcting mistakes, filling gaps, strengthening justification, clarifying assumptions or conditions, or reusing promising ideas from the draft, or by doing several of these at once.

{previous_answer}

## Final Instruction
Your final response should follow the format above, including an '## Solution' section.
"""