import os
import re
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

from openai import OpenAI


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
    max_tokens: int = 32768  # Increased slightly for detailed critiques


class ProofVerifier:
    """
    Proof verifier that supports standard (single-pass), pessimistic (multi-pass),
    and progressive (hierarchical chunking) verification strategies.
    """

    def __init__(self, config: ProofVerifierConfig):
        self.config = config
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY_KEY"
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def _split_into_chunks(self, proof: str, chunk_length: int) -> List[str]:
        """Split proof into chunks of N lines."""
        lines = (proof or "").splitlines()
        if not lines:
            return [proof or ""]
        chunks = []
        for i in range(0, len(lines), chunk_length):
            chunk_lines = lines[i : i + chunk_length]
            chunks.append("\n".join(chunk_lines))
        return chunks

    def _calculate_chunk_length(self, proof: str, iteration: int, min_chunk_size: int = 6) -> int:
        """Calculate dynamic chunk length based on iteration depth."""
        lines = (proof or "").splitlines()
        num_lines = len(lines)
        if num_lines == 0:
            return min_chunk_size
        
        # Iteration 0 is always full proof (handled externally usually, but logic holds)
        if iteration == 0:
            return max(num_lines, min_chunk_size)
            
        # Progressive granularization: 2, 4, 8... chunks
        target_chunks = max(1, 2**iteration)
        approx_length = math.ceil(num_lines / target_chunks)
        return max(min_chunk_size, approx_length)

    def _build_standard_prompt(self, problem: str, proof: str) -> List[Dict[str, str]]:
        """Compose the standard full-proof verification prompt."""
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
        """Compose the prompt for verifying a specific chunk within the context of the full proof."""
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

    def _call_model(self, messages: List[Dict[str, str]]) -> tuple[bool, str]:
        """Perform the actual API call."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content or ""
            verdict_text = extract_xml_content(content, "verification")
            
            # Default to False if tag is missing to be safe, or True? 
            # Usually safe to assume False if format is broken in verification tasks.
            verdict = verdict_text.strip().lower() == "true" if verdict_text else False
            return verdict, content
        except Exception as e:
            # Fallback for API errors
            return False, f"API Error: {str(e)}"

    def _verify_progressive(self, problem: str, proof: str, max_iters: int = 3) -> Dict[str, Any]:
        """
        Iteratively verify the proof starting with a full pass, then breaking into
        progressively smaller chunks (hierarchical review).
        """
        all_reviews = []
        
        # Iteration 0: Standard Full Review
        verdict, review = self._call_model(self._build_standard_prompt(problem, proof))
        all_reviews.append(f"[Full Pass] {review}")
        
        if not verdict:
            return {
                "verdict": False,
                "review": review,
                "all_reviews": all_reviews,
                "strategy": "progressive_chunked"
            }

        # Iteration 1 to max_iters: Chunked Reviews
        # We start iteration 1 (2 chunks roughly) up to max_iters.
        # If any chunk fails, the whole proof is marked incorrect.
        for i in range(1, max_iters):
            chunk_len = self._calculate_chunk_length(proof, i)
            chunks = self._split_into_chunks(proof, chunk_len)
            
            # Review all chunks for this iteration level
            # Note: In a production async environment, these would run in parallel.
            for idx, chunk in enumerate(chunks, 1):
                msgs = self._build_chunk_prompt(problem, proof, chunk, idx)
                c_verdict, c_review = self._call_model(msgs)
                
                # We log the review internally
                all_reviews.append(f"[Iter {i} Chunk {idx}] {c_review}")
                
                if not c_verdict:
                    return {
                        "verdict": False,
                        "review": f"Error found in Iteration {i}, Chunk {idx}:\n{c_review}",
                        "all_reviews": all_reviews,
                        "strategy": "progressive_chunked"
                    }

        # If we survive all iterations
        return {
            "verdict": True,
            "review": all_reviews[-1],  # Return the last successful review
            "all_reviews": all_reviews,
            "strategy": "progressive_chunked"
        }

    def verify(self, problem: str, proof: str, reviewer: str = "standard", reviews: int = 3):
        """
        Verify a proof using:
            - standard: single review.
            - pessimistic: run up to `reviews` independent standard reviews; stop at first failure.
            - progressive: hierarchical chunking strategy (Iterative deepening).
        """
        review_texts: List[str] = []
        
        if reviewer == "progressive":
            # Map 'progressive' to the new hierarchical chunking logic
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
):
    """
    Run proof verification and format the result.
    
    Args:
        proof_output: The solution text to verify.
        problem: The problem text.
        reviewer: 'standard', 'pessimistic', or 'progressive'.
        reviews: Number of reviews for pessimistic or iterations for progressive.
    """
    config = ProofVerifierConfig(
        model_name=model_name,
        base_url=f"http://127.0.0.1:{model_port}/v1",
        api_key=api_key,
    )
    verifier = ProofVerifier(config)
    result = verifier.verify(problem, proof_output, reviewer=reviewer, reviews=reviews)

    score = 1.0 if result["verdict"] else 0.0
    return {
        "score": score,
        "point": score,
        "acc": bool(result["verdict"]),
        "extracted_gt": "",
        "extracted_pred": "",
        "scored_by": "proof",
        "score_noxverify": score,
        "point_noxverify": score,
    }