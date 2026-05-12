import os
import re
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

from openai import OpenAI

# 定义细粒度评估 Prompt 模板 (Naive 模式)
# 移除了 Reference Solution 和 Guidelines，仅保留 Problem 和 Student Response
PROMPT_FINE_GRAINED = """
You are an expert Mathematics Olympiad grader using a **Dual-Verification Protocol**. 
Your task is to be both empathetic to the student's strategy and ruthlessly logical in its verification.

### 1. Grading Categories (Official)
- **Correct** (7 points): The method is valid and the logic survives both forward and backward verification.
- **Almost** (6 points): Logically sound but contains minor, non-fatal errors.
- **Partial** (1 point): Contains a key insight listed in the guidelines, but the main proof fails.
- **Incorrect** (0 points): The logic is fundamentally flawed, circular, or contains non-reversible errors.

### 2. Practical Tips (Be Open-Minded)
1. **Alternative Methods are Valid:** If the student's method is different from the reference, evaluate its own internal logic.
2. **Check for Equivalence:** Different mathematical forms can be equally correct.

### 3. Critical Reality Checks (Be Skeptical)
1. **Do Not 'Fix' Major Gaps:** Do not invent missing core arguments for the student.
2. **Challenge 'Hand-Waving':** Phrases like 'it is obvious' often hide the most difficult part of the proof. Verify these claims.

### 4. Problem Context
**Problem:**
{problem}

### 5. Student Response to Grade
{student_response}

### 6. Dual-Verification Protocol (CRUCIAL)
You must perform BOTH checks before making a final decision:

**Check 1: Forward Trace (Follow the Path)**
- Start from the student's initial assumptions.
- Follow their derivation step-by-step. Ask: 'Does step N logically follow from step N-1?'
- This check ensures the flow of logic is valid.

**Check 2: Backward Verification (Challenge the Path)**
- Start from the student's **final conclusion**.
- Work backward, step-by-step, towards their premises. Ask: 'Is the previous step a necessary and sufficient condition for the current step?'
- This check is EXTREMELY effective at catching **non-reversible steps** (e.g., `x=2 => x^2=4`, but `x^2=4` does not imply `x=2`) and hidden assumptions.

**Synthesis:**
- If both checks pass, the logic is likely **Correct**.
- If the forward check passes but the backward check fails, a non-reversible error was likely made. Grade as **Incorrect**.
- Use this dual perspective to assign the final grade.

### 7. Final Output
End your response strictly with:
Final Answer: <category>
(Where <category> is strictly one of: correct, almost, partial, incorrect)
"""

# 定义 Marking 模式的 Prompt 模板
# 根据 marking 标准逐步评分，总分固定为 7 分（模仿 IMO 题目设计）
PROMPT_MARKING = """
You are an expert Mathematics Olympiad grader. Your task is to evaluate a student's solution based on a detailed marking scheme.

**Important:** This is an IMO-style problem with a maximum score of 7 points.

### 1. Problem
{problem}

### 2. Student Response
{student_response}

### 3. Marking Scheme (Total: 7 points)
The following marking scheme lists the criteria and corresponding points for each step. The points should sum to 7. Evaluate the student's solution against each criterion:

{marking_scheme}

### 4. Grading Instructions
For each criterion in the marking scheme:
1. Carefully check if the student's solution satisfies the criterion.
2. Award the FULL points for that criterion if it is completely satisfied.
3. Award 0 points if the criterion is NOT satisfied (no partial points within a single criterion).
4. Be open-minded about alternative approaches - if the student uses a different but valid method that achieves the same goal described in the criterion, award the points.
5. Do NOT invent or assume steps that are not explicitly present in the solution.

### 5. Output Format
For each criterion, output your evaluation in the following format:
<criterion_N>
- Criterion: [Quote the criterion]
- Max Points: [Points available]
- Satisfied: [Yes/No]
- Awarded: [Points awarded]
- Reason: [Brief justification]
</criterion_N>

At the end, provide the total score out of 7:
<total_score>X.X</total_score>

Where X.X is the sum of all awarded points (0 to 7).
"""

def extract_xml_content(text: str, tag: str) -> Optional[str]:
    """Extract the last occurrence of <tag>content</tag> from a string."""
    pattern = rf"<{re.escape(tag)}(?:\s+[^>]*)?\s*>(.*?)</\s*{re.escape(tag)}\s*>"
    matches = list(re.finditer(pattern, text or "", flags=re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def strip_think_simple(content: str) -> str:
    """Remove <think> blocks often returned by reasoning models."""
    return re.sub(r"<think\b[^>]*>.*?</think>", "", content or "", flags=re.DOTALL | re.IGNORECASE)


@dataclass
class ProofVerifierConfig:
    model_name: str = "gpt-oss-120b"
    base_url: str = "http://127.0.0.1:34882/v1"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 32768


class ProofVerifier:
    def __init__(self, config: ProofVerifierConfig):
        self.config = config
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY_KEY"
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def _split_into_chunks(self, proof: str, chunk_length: int) -> List[str]:
        lines = (proof or "").splitlines()
        if not lines:
            return [proof or ""]
        chunks = []
        for i in range(0, len(lines), chunk_length):
            chunk_lines = lines[i : i + chunk_length]
            chunks.append("\n".join(chunk_lines))
        return chunks

    def _calculate_chunk_length(self, proof: str, iteration: int, min_chunk_size: int = 6) -> int:
        lines = (proof or "").splitlines()
        num_lines = len(lines)
        if num_lines == 0:
            return min_chunk_size
        
        if iteration == 0:
            return max(num_lines, min_chunk_size)
            
        target_chunks = max(1, 2**iteration)
        approx_length = math.ceil(num_lines / target_chunks)
        return max(min_chunk_size, approx_length)

    def _build_standard_prompt(self, problem: str, proof: str) -> List[Dict[str, str]]:
        cleaned_proof = strip_think_simple(proof)
        return [
            {
                "role": "system",
                "content": (
                    "You are an assistant highly proficient in mathematics. "
                    "The user will provide a math problem together with its proposed solution, "
                    "and your task is to verify the correctness of that solution according to the given instruction."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here is a math problem and a candidate solution of it, and you need to verify the correctness "
                    "of this solution. Please check each of the following:\n\n"
                    "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
                    "2. The solution actually derives the conclusion required by the original problem.\n"
                    "3. Every step of calculation and formula derivation in the solution is correct.\n"
                    "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
                    "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n\n"
                    "Then, conduct Backward Verification (Challenge the Path). Please use the following step:\n"
                    "1. Start from the student's **final conclusion**.\n"
                    "2. Work backward, step-by-step, towards their premises. Ask: 'Is the previous step a necessary and sufficient condition for the current step?'\n"
                    "3. This check is EXTREMELY effective at catching **non-reversible steps** (e.g., `x=2 => x^2=4`, but `x^2=4` does not imply `x=2`) and hidden assumptions.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the solution as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness (e.g., invalid step, wrong theorem usage without required conditions, uncorrected calculation error leading to a wrong result), treat the solution as incorrect.\n\n"
                    "Response requirements: If the solution is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any. "
                    "If the solution is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error. "
                    "Do not include any restatement of the entire solution or problem.\n\n"
                    f"<problem>{problem}</problem>\n\n"
                    f"<answer>{cleaned_proof}</answer>"
                ),
            },
        ]

    def _build_chunk_prompt(self, problem: str, full_proof: str, chunk: str, chunk_idx: int) -> List[Dict[str, str]]:
        cleaned_proof = strip_think_simple(full_proof)
        return [
            {
                "role": "system",
                "content": (
                    "You are an assistant highly proficient in mathematics. "
                    "The user will provide a math problem together with its proposed solution, "
                    "and your task is to verify the correctness of that solution."
                ),
            },
            {
                "role": "user",
                "content": (
                    "We provide the original problem and the complete proposed solution for full context. "
                    "Then we provide a specific chunk from the solution for focused checking. "
                    "Your task: Check ONLY the given chunk for errors while considering the overall context.\n\n"
                    "Checklist:\n"
                    "1. The chunk's reasoning and calculations adhere to mathematical correctness.\n"
                    "2. Any theorems used in the chunk match their hypotheses and conclusions.\n"
                    "3. The chunk does not rely on assumptions not justified by the problem or earlier proven steps.\n\n"
                    "Then, conduct Backward Verification (Challenge the Path). Please use the following step:\n"
                    "1. Start from the student's **final conclusion**.\n"
                    "2. Work backward, step-by-step, towards their premises. Ask: 'Is the previous step a necessary and sufficient condition for the current step?'\n"
                    "3. This check is EXTREMELY effective at catching **non-reversible steps** (e.g., `x=2 => x^2=4`, but `x^2=4` does not imply `x=2`) and hidden assumptions.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the chunk as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness in this chunk (e.g., invalid step, wrong theorem usage without required conditions), treat the chunk as incorrect.\n\n"
                    "Response requirements: If the chunk is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any. "
                    "If the chunk is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error in the proof that you found in the chunk.\n\n"
                    f"<problem>{problem}</problem>\n\n"
                    f"<full_answer>{cleaned_proof}</full_answer>\n\n"
                    f"<chunk_index>{chunk_idx}</chunk_index>\n"
                    f"<chunk>{chunk}</chunk>"
                ),
            },
        ]
    
    def _build_naive_prompt(self, problem: str, proof: str) -> List[Dict[str, str]]:
        """Compose the naive fine-grained grading prompt without ref solution/guidelines."""
        cleaned_proof = strip_think_simple(proof)
        prompt_content = PROMPT_FINE_GRAINED.format(
            problem=problem,
            student_response=cleaned_proof
        )
        return [
            {"role": "user", "content": prompt_content}
        ]

    def _build_marking_prompt(self, problem: str, proof: str, marking: List[str]) -> List[Dict[str, str]]:
        """Compose the marking-based grading prompt with detailed scoring criteria."""
        cleaned_proof = strip_think_simple(proof)
        
        # 将 marking 列表格式化为编号的评分标准
        marking_scheme_lines = []
        for i, criterion in enumerate(marking, 1):
            marking_scheme_lines.append(f"Criterion {i}: {criterion}")
        marking_scheme = "\n".join(marking_scheme_lines)
        
        prompt_content = PROMPT_MARKING.format(
            problem=problem,
            student_response=cleaned_proof,
            marking_scheme=marking_scheme
        )
        return [
            {"role": "user", "content": prompt_content}
        ]

    def _call_model(self, messages: List[Dict[str, str]], mode: str = "standard") -> tuple[Any, str]:
        """Perform the actual API call. Return (verdict/score, content)."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content or ""
            
            if mode == "naive":
                # 解析 Naive 模式的 Final Answer
                last_line = ""
                # 尝试找最后一行包含 Final Answer 的
                for line in reversed(content.split('\n')):
                    if "final answer:" in line.lower():
                        last_line = line.lower()
                        break
                
                score = 0.0
                # Mapping logic: Correct=7, Almost=6, Partial=1, Incorrect=0
                # Normalized by /7: 1.0, 0.857, 0.143, 0.0
                if "correct" in last_line and "incorrect" not in last_line:
                    score = 1.0 
                elif "almost" in last_line:
                    score = 6/7
                elif "partial" in last_line:
                    score = 1/7
                elif "incorrect" in last_line:
                    score = 0.0
                else:
                    # Fallback keyword search
                    lower_content = content.lower()
                    if "final answer: correct" in lower_content: score = 1.0
                    elif "final answer: almost" in lower_content: score = 6/7
                    elif "final answer: partial" in lower_content: score = 1/7
                    elif "final answer: incorrect" in lower_content: score = 0.0
                
                return score, content

            elif mode == "marking":
                # 解析 Marking 模式的分数（IMO 风格，总分固定为 7 分）
                total_score_text = extract_xml_content(content, "total_score")
                max_score = 7.0  # IMO 题目总分固定为 7 分
                
                try:
                    total_score = float(total_score_text) if total_score_text else 0.0
                    # 确保分数在 [0, 7] 范围内
                    total_score = max(0.0, min(7.0, total_score))
                    # 归一化分数到 0-1 范围
                    normalized_score = total_score / max_score
                except (ValueError, TypeError):
                    normalized_score = 0.0
                    total_score = 0.0
                
                return {
                    "normalized_score": normalized_score,
                    "total_score": total_score,
                    "max_score": max_score
                }, content

            else:
                # Standard XML verification parsing
                verdict_text = extract_xml_content(content, "verification")
                verdict = verdict_text.strip().lower() == "true" if verdict_text else False
                return verdict, content

        except Exception as e:
            if mode == "naive":
                return 0.0, f"API Error: {str(e)}"
            elif mode == "marking":
                return {"normalized_score": 0.0, "total_score": 0.0, "max_score": 7.0}, f"API Error: {str(e)}"
            return False, f"API Error: {str(e)}"

    def _verify_progressive(self, problem: str, proof: str, max_iters: int = 3) -> Dict[str, Any]:
        all_reviews = []
        verdict, review = self._call_model(self._build_standard_prompt(problem, proof))
        all_reviews.append(f"[Full Pass] {review}")
        
        if not verdict:
            return {
                "verdict": False,
                "review": review,
                "all_reviews": all_reviews,
                "strategy": "progressive_chunked"
            }

        for i in range(1, max_iters):
            chunk_len = self._calculate_chunk_length(proof, i)
            chunks = self._split_into_chunks(proof, chunk_len)
            
            for idx, chunk in enumerate(chunks, 1):
                msgs = self._build_chunk_prompt(problem, proof, chunk, idx)
                c_verdict, c_review = self._call_model(msgs)
                all_reviews.append(f"[Iter {i} Chunk {idx}] {c_review}")
                
                if not c_verdict:
                    return {
                        "verdict": False,
                        "review": f"Error found in Iteration {i}, Chunk {idx}:\n{c_review}",
                        "all_reviews": all_reviews,
                        "strategy": "progressive_chunked"
                    }

        return {
            "verdict": True,
            "review": all_reviews[-1],
            "all_reviews": all_reviews,
            "strategy": "progressive_chunked"
        }

    def verify(self, problem: str, proof: str, reviewer: str = "standard", reviews: int = 3, marking: Optional[List[str]] = None):
        """
        Verify a proof using specified strategy.
        
        Args:
            problem: The problem statement
            proof: The student's proof/solution
            reviewer: The review strategy ("standard", "naive", "progressive", "pessimistic", "marking")
            reviews: Number of reviews for multi-pass strategies
            marking: Optional list of marking criteria for "marking" mode
        """
        review_texts: List[str] = []
        
        # 如果提供了 marking，自动切换到 marking 模式
        if marking is not None and len(marking) > 0:
            reviewer = "marking"
        
        if reviewer == "marking":
            if marking is None or len(marking) == 0:
                raise ValueError("marking list is required for 'marking' reviewer mode")
            
            # Marking 模式：根据详细评分标准打分（只进行单次验证）
            msgs = self._build_marking_prompt(problem, proof, marking)
            score_result, review = self._call_model(msgs, mode="marking")
            
            return {
                "verdict": score_result["normalized_score"] > 0.0,
                "score": score_result["normalized_score"],
                "total_score": score_result["total_score"],
                "max_score": score_result["max_score"],
                "review": review,
                "all_reviews": [review],
                "strategy": "marking",
                "reviews_ran": 1
            }
        
        elif reviewer == "naive":
            # Naive / Fine-grained grading mode
            # 移除了 ref_solution 和 guidelines 的传递
            msgs = self._build_naive_prompt(problem, proof)
            score, review = self._call_model(msgs, mode="naive")
            
            return {
                "verdict": score > 0.0, # 只要不是纯 incorrect 就算作有分数 (或者根据需要调整阈值)
                "score": score,         # 归一化后的分数 (0 ~ 1)
                "review": review,
                "all_reviews": [review],
                "strategy": "naive",
                "reviews_ran": 1
            }

        elif reviewer == "progressive":
            return self._verify_progressive(problem, proof, max_iters=reviews)

        elif reviewer == "pessimistic":
            total_reviews = max(1, int(reviews) if reviews else 1)
            verdict = True
            final_review = ""
            
            for i in range(total_reviews):
                verdict, review = self._call_model(self._build_standard_prompt(problem, proof))
                review_texts.append(f"[Run {i+1}] {review}")
                if not verdict:
                    final_review = review
                    break
                final_review = review
            
            return {
                "verdict": verdict,
                "score": 1.0 if verdict else 0.0,
                "review": final_review,
                "all_reviews": review_texts,
                "strategy": reviewer,
                "reviews_ran": len(review_texts),
            }
            
        else:
            # Standard single pass
            verdict, review = self._call_model(self._build_standard_prompt(problem, proof))
            return {
                "verdict": verdict,
                "score": 1.0 if verdict else 0.0,
                "review": review,
                "all_reviews": [review],
                "strategy": "standard",
                "reviews_ran": 1,
            }


def compute_score_proof(
    proof_output: str,
    problem: str,
    reviewer: str = "standard",
    reviews: int = 3,
    model_port: int = 34882,
    model_name: str = "gpt-oss-120b",
    api_key: Optional[str] = None,
    marking: Optional[List[str]] = None,
):
    """
    Run proof verification and format the result.
    
    Args:
        proof_output: The student's proof/solution
        problem: The problem statement
        reviewer: The review strategy ("standard", "naive", "progressive", "pessimistic", "marking")
        reviews: Number of reviews for multi-pass strategies
        model_port: Port for the verifier model API
        model_name: Name of the verifier model
        api_key: Optional API key
        marking: Optional list of marking criteria. If provided, automatically uses "marking" mode.
    """
    config = ProofVerifierConfig(
        model_name=model_name,
        base_url=f"http://127.0.0.1:{model_port}/v1",
        api_key=api_key,
    )
    verifier = ProofVerifier(config)
    result = verifier.verify(
        problem, 
        proof_output, 
        reviewer=reviewer, 
        reviews=reviews,
        marking=marking,
    )

    # Use the calculated score from naive/marking mode, or 1/0 from binary modes
    score = result.get("score", 1.0 if result["verdict"] else 0.0)
    
    response = {
        "score": score,
        "point": score,
        "acc": bool(result["verdict"]),
        "extracted_gt": "",
        "extracted_pred": "",
        "scored_by": "proof",
        "score_noxverify": score,
        "point_noxverify": score,
        "review": result.get("review", "")
    }
    
    # 如果是 marking 模式，添加额外的得分信息（IMO 风格，总分 7 分）
    if result.get("strategy") == "marking":
        response["total_score"] = result.get("total_score", 0.0)
        response["max_score"] = result.get("max_score", 7.0)
        response["scored_by"] = "proof_marking"
    
    return response