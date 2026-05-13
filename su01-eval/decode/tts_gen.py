"""
MIT License

Copyright (c) 2025 Lin Yang, Yichen Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import re
import sys
import json
import requests
import argparse
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- CONFIGURATION ---
# OpenAI-compatible /v1/chat/completions endpoint configuration.
MODEL_NAME = os.environ.get("MODEL_NAME", "SU01")
API_URL = os.environ.get("API_URL", "http://localhost:34883/v1/chat/completions")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "160000"))
_TOP_K_RAW = os.environ.get("TOP_K")
TOP_K = int(_TOP_K_RAW) if _TOP_K_RAW not in (None, "") else -1
TOP_P = float(os.environ.get("TOP_P", "0.95"))
API_TIMEOUT = float(os.environ.get("API_TIMEOUT", "432000000"))
DEFAULT_NO_PROXY = "localhost,127.0.0.1"


def configure_proxy_env():
    """Match the default eval environment for direct service calls."""
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [part.strip() for part in existing.split(",") if part.strip()]
    for part in DEFAULT_NO_PROXY.split(","):
        if part not in parts:
            parts.append(part)
    no_proxy = ",".join(parts)
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(name, None)


def _env_truthy(name, default="0"):
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


configure_proxy_env()
DECODE_DRY_RUN = _env_truthy("DECODE_DRY_RUN")
# Global variables for logging

# Verification parameters. Keep true + false thresholds within exploration rounds.
MAX_VERIFICATION_TRUE_ROUNDS = 5
MAX_VERIFICATION_FALSE_ROUNDS = 10
MAX_EXPLORATION_ROUNDS = 30
PROOF_DATASETS = {"imo24", "imo25", "proofbench"}

_log_file = None
_print_lock = threading.Lock()
original_print = print

def log_print(*args, **kwargs):
    """Thread-safe print function that mirrors output to log file."""
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    flush = kwargs.get("flush", False)
    file_obj = kwargs.get("file", None)

    # Keep explicit `file=` behavior unchanged.
    if file_obj is not None:
        return original_print(*args, **kwargs)

    message = sep.join(str(arg) for arg in args)

    # Add timestamp to lines starting with ">>>>>"
    if message.startswith('>>>>>'):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[{timestamp}] {message}"

    # Serialize stdout + file writes to avoid TextIO race corruption (NUL bytes).
    with _print_lock:
        original_print(message, end=end, flush=flush)
        if _log_file is not None:
            _log_file.write(message + end)
            _log_file.flush()

# Replace the built-in print function
print = log_print

def _requests_proxies_for_url(url):
    """Bypass HTTP(S) proxies for localhost and RFC1918 API endpoints."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return None
    if not host:
        return None
    bypass = {"http": None, "https": None}
    if host in ("localhost", "127.0.0.1", "::1"):
        return bypass
    if host.startswith("10."):
        return bypass
    if host.startswith("192.168."):
        return bypass
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            n = int(parts[1])
            if 16 <= n <= 31:
                return bypass
    return None

def set_log_file(log_file_path):
    """Set the log file for output."""
    global _log_file
    if log_file_path:
        try:
            log_dir = os.path.dirname(os.path.abspath(log_file_path))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            if _log_file is not None:
                _log_file.close()
            _log_file = open(log_file_path, 'w', encoding='utf-8', buffering=1)
            return True
        except Exception as e:
            print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True

def close_log_file():
    """Close the log file if it's open."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

step1_prompt = """
### Core Instructions ###

*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    *   Proving a key lemma.
    *   Fully resolving one or more cases within a logically sound case-based proof.
    *   Establishing a critical property of the mathematical objects in the problem.
    *   For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings. This section must contain two parts:

*   **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
    *   **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
*   **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
    *   A narrative of your overall strategy.
    *   The full and precise mathematical statements of any key lemmas or major intermediate results.
    *   If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.

"""

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt.
"""

check_verification_prompt = """
Can you carefully review each item in your list of findings? Are they valid or overly strict? An expert grader must be able to distinguish between a genuine flaw and a concise argument that is nonetheless sound, and to correct their own assessment when necessary.

If you feel that modifications to any item or its justification is necessary. Please produce a new list. In your final output, please directly start with **Summary** (no need to justify the new list).
"""

correction_prompt = """
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.
"""

parse_reformat_prompt = """
Parsing/format issue detected:
- Your previous response could not be parsed into a valid solution body.
- Please regenerate the full solution.
- The output MUST contain BOTH sections, in this order:
  1. Summary
  2. Detailed Solution
- Put the complete proof under "Detailed Solution".
- Do NOT output grader-style text such as "Final Verdict" or "Detailed Verification Log".
"""

verification_system_prompt = """
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
*   Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
*   You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

*   **a. Critical Error:**
    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
    *   **Procedure:**
        *   Explain the specific error and state that it **invalidates the current line of reasoning**.
        *   Do NOT check any further steps that rely on this error.
        *   You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

*   **b. Justification Gap:**
    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
    *   **Procedure:**
        *   Explain the gap in the justification.
        *   State that you will **assume the step's conclusion is true** for the sake of argument.
        *   Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

*   **a. Summary**
    This section MUST be at the very beginning of your response. It must contain two components:
    *   **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
    *   **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
        *   **Location:** A direct quote of the key phrase or equation where the issue occurs.
        *   **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

*   **b. Detailed Verification Log**
    Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.

**Example of the Required Summary Format**
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*

**Final Verdict:** The solution is **invalid** because it contains a Critical Error.

**List of Findings:**
*   **Location:** "By interchanging the limit and the integral, we get..."
    *   **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
*   **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"
    *   **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.

"""


verification_remider = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""

def get_api_key():
    """
    Retrieves the OpenAI API key from environment variables.
    Exits if the key is not found.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if DECODE_DRY_RUN and not api_key:
        return "dummy"
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set the variable, e.g., 'export OPENAI_API_KEY=\"your_api_key\"'")
        sys.exit(1)
    return api_key

def read_file_content(filepath):
    """
    Reads and returns the content of a file.
    Exits if the file cannot be read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def build_request_payload(system_prompt, question_prompt, other_prompts=None):
    """
    Builds the JSON payload for vLLM /v1/chat/completions.
    Returns a dict with a 'messages' list for proper multi-turn handling.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = question_prompt
    if other_prompts:
        for prompt in other_prompts:
            user_content += f"\n\nAdditional instruction: {prompt}"
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
    }
    if TOP_K is not None:
        payload["top_k"] = TOP_K
    return payload

def send_api_request(api_key, payload):
    """
    Sends the request to the OpenAI API and returns the response.
    """
    if DECODE_DRY_RUN:
        return _build_dry_run_response(payload)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = None
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=API_TIMEOUT,
            proxies=_requests_proxies_for_url(API_URL),
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        err_resp = getattr(e, "response", None) or response
        if err_resp is not None:
            print(f"HTTP status: {err_resp.status_code}")
            if err_resp.status_code == 400:
                print(f"Possible reason for 400: model route mismatch, invalid sampling params, or request too long.")
            print(f"Raw API Response: {err_resp.text}")
        else:
            print("No HTTP response body available (request may have failed before reaching server).")
        raise e


def _build_dry_run_response(payload):
    messages = payload.get("messages") or []
    joined_prompt = "\n\n".join(str(m.get("content", "")) for m in messages)
    lower_prompt = joined_prompt.lower()

    if "final_answer: yes" in lower_prompt and "final_answer: no" in lower_prompt:
        content = "FINAL_ANSWER: yes"
    elif "verification task reminder" in lower_prompt or "detailed verification log" in lower_prompt:
        content = (
            "**Final Verdict:** The solution is correct.\n\n"
            "**List of Findings:** None.\n\n"
            "**Detailed Verification Log**\n"
            "Dry-run verification accepts the deterministic mock solution."
        )
    elif "the following is a detailed mathematical solution" in lower_prompt:
        content = "[DRY_RUN] formatted final answer."
    else:
        content = (
            "**1. Summary**\n"
            "Dry-run generated a deterministic mock solution for smoke testing.\n\n"
            "**2. Detailed Solution**\n"
            "This response is produced without contacting API_URL."
        )

    return {
        "dry_run": True,
        "choices": [{"message": {"content": content}}],
    }

def strip_thinking(content: str) -> str:
    """
    Strip <think>...</think> reasoning markup when present.
    """
    if '</think>' in content:
        return content.split('</think>', 1)[-1].lstrip('\n')
    return content


def extract_text_from_response(response_data):
    """
    Extracts the generated text from a vLLM /v1/chat/completions response.
    Strips <think>...</think> reasoning content, returning only the answer.
    """
    try:
        print(">>>>>> Response:")
        print(json.dumps(response_data, indent=2))

        content = response_data['choices'][0]['message']['content']
        return strip_thinking(content)
    except (KeyError, IndexError, TypeError) as e:
        print("Error: Could not extract text from the API response.")
        print(f"Reason: {e}")
        print("Full API Response:")
        print(json.dumps(response_data, indent=2))
        raise e

_PARSE_FAIL_TAG = "[AUTO_PARSE_ERROR]"

_DETAILED_SOLUTION_HEADING_PATTERNS = [
    # e.g. "## 2. Detailed Solution" / "### Detailed Solution ###"
    re.compile(
        r'(?im)^\s{0,3}#{1,6}\s*(?:\d+[\.\)]\s*)?Detailed\s+Solution(?:\s*[:：])?\s*(?:#{1,6})?\s*$'
    ),
    # e.g. "**2. Detailed Solution**"
    re.compile(
        r'(?im)^\s{0,3}\*{0,2}\s*(?:\d+[\.\)]\s*)?Detailed\s+Solution(?:\s*[:：])?\s*\*{0,2}\s*$'
    ),
]

_SOLUTION_HEADING_PATTERNS = [
    # e.g. "## Solution" / "### Solution ###"
    re.compile(
        r'(?im)^\s{0,3}#{1,6}\s*(?:\d+[\.\)]\s*)?(?:Final\s+)?Solution(?:\s*[:：])?\s*(?:#{1,6})?\s*$'
    ),
    re.compile(
        r'(?im)^\s{0,3}\*{0,2}\s*(?:\d+[\.\)]\s*)?(?:Final\s+)?Solution(?:\s*[:：])?\s*\*{0,2}\s*$'
    ),
]


def _extract_after_heading_patterns(text: str, patterns):
    """Try a list of heading regexes and return text after the first match."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            return text[m.end():].strip(), pat.pattern
    return "", ""


def extract_detailed_solution(solution, marker='Detailed Solution', after=True):
    """
    Robustly extract solution body after a "Detailed Solution" / "Solution" section heading.
    Keeps backward compatibility with the previous marker-based behavior.
    """
    text = (solution or "")
    if not text:
        return ""

    if not after:
        idx = text.lower().find(marker.lower())
        if idx == -1:
            return ""
        return text[:idx].strip()

    body, _ = _extract_after_heading_patterns(text, _DETAILED_SOLUTION_HEADING_PATTERNS)
    if body:
        return body

    # Backward-compatible fallback for plain marker substring match.
    idx = text.lower().find(marker.lower())
    if idx != -1:
        tail = text[idx + len(marker):].strip()
        if tail:
            return tail

    # If model outputs "## Solution" (common in logs), still extract that body.
    body, _ = _extract_after_heading_patterns(text, _SOLUTION_HEADING_PATTERNS)
    if body:
        return body

    return ""


def extract_solution_for_verification(solution):
    """
    Extract solution text for verifier:
    - Prefer parsed section body.
    - If parsing fails but output is non-empty, fallback to full text instead of empty.
    """
    text = (solution or "").strip()
    if not text:
        return "", "empty_output"

    dsol = extract_detailed_solution(text)
    if dsol:
        return dsol, "section_extracted"

    return text, "fallback_full_text"


def _looks_like_invalid_solution_payload(dsol: str):
    """
    Detect obvious non-solution payloads to avoid sending empty/garbled content to verifier.
    """
    text = (dsol or "").strip()
    if not text:
        return True, "empty_solution_body"

    lower = text.lower()
    if re.match(r'^\s*###\s*verification task reminder\s*###?(?:\s|$)', text, re.IGNORECASE):
        return True, "solution_body_is_verification_reminder"
    if lower.startswith("we need to evaluate the provided solution"):
        return True, "model_meta_reasoning_instead_of_solution"
    if "detailed verification log" in lower and "final verdict" in lower:
        return True, "verifier_style_output_instead_of_solution"

    return False, ""


def _build_parse_failure_bug_report(solution: str, parse_source: str, reason: str) -> str:
    preview = (solution or "").strip().replace("\n", "\\n")
    preview = preview[:600]
    return (
        f"{_PARSE_FAIL_TAG} Could not extract a valid solution body for verification. "
        f"source={parse_source}; reason={reason}. "
        "Please regenerate and include an explicit 'Detailed Solution' section.\n"
        f"Raw output preview: {preview}"
    )


def is_parse_failure_report(report: str) -> bool:
    return isinstance(report, str) and report.startswith(_PARSE_FAIL_TAG)


def parse_verdict(o: str) -> str:
    """
    Parse the final yes/no verdict from a model response.

    Priority:
      1. "FINAL_ANSWER: yes/no".
      2. Common conclusion sentences such as "the answer is yes/no".
      3. A line containing only yes/no.
      4. The last standalone yes/no token.
      5. Fall back to "no" conservatively.
    """
    text = o.strip()

    m = re.search(r'FINAL_ANSWER\s*:\s*(yes|no)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    conclusion_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:：]?\s*(yes|no)\b',
        r'thus\s+(?:the\s+)?answer\s+(?:is\s+)?(yes|no)\b',
        r'(?:so\s+)?answer\s*[:：]\s*(yes|no)\b',
        r'(?:my\s+)?(?:final\s+)?verdict\s+is\s*(yes|no)\b',
        r'(?:conclusion|result)\s*[:：]\s*(yes|no)\b',
        r'(?:is\s+(?:therefore|hence|thus)\s+)?(yes|no)\b',
    ]
    all_matches = []
    for pat in conclusion_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            all_matches.append((m.start(), m.group(1).lower()))
    if all_matches:
        all_matches.sort(key=lambda x: x[0])
        return all_matches[-1][1]

    for line in reversed(text.splitlines()):
        stripped = line.strip().rstrip('.,!').strip().lower()
        if stripped in ('yes', 'no'):
            return stripped

    for m in reversed(list(re.finditer(r'\b(yes|no)\b', text, re.IGNORECASE))):
        return m.group(1).lower()

    print("[parse_verdict] WARNING: could not parse yes/no from response, defaulting to 'no'")
    return "no"


# Case-insensitive markers used to split summary from detailed verification.
_BUG_REPORT_SPLIT_MARKERS = [
    "Detailed Verification Log",
    "Detailed Verification",
    "Detailed verification log",
    "Detailed verification",
    "Step-by-step verification",
    "Verification Log",
]


def extract_bug_report(out: str) -> str:
    """
    Extract the summary/bug-report portion from a verifier response.

    Falls back to the full response if no known split marker is found.
    """
    out_lower = out.lower()
    for marker in _BUG_REPORT_SPLIT_MARKERS:
        idx = out_lower.find(marker.lower())
        if idx != -1:
            return out[:idx].strip()

    print("[extract_bug_report] WARNING: no split marker found, using full verification output as bug_report")
    return out.strip()


def verify_solution(problem_statement, solution, verbose=True, stop_event=None):
    if stop_event is not None and stop_event.is_set():
        if verbose:
            print("[verify_solution] Skip verification because early-stop signal is set.")
        return "", "no"

    dsol, parse_source = extract_solution_for_verification(solution)
    is_invalid_payload, invalid_reason = _looks_like_invalid_solution_payload(dsol)

    if is_invalid_payload:
        bug_report = _build_parse_failure_bug_report(solution, parse_source, invalid_reason)
        if(verbose):
            print("[verify_solution] Skip verifier due to parse/format issue.")
            print(json.dumps(bug_report, indent=4))
        return bug_report, "no"

    if(verbose and parse_source != "section_extracted"):
        print(f"[verify_solution] WARNING: using {parse_source} for verification payload.")

    if stop_event is not None and stop_event.is_set():
        if verbose:
            print("[verify_solution] Interrupted before verifier request.")
        return "", "no"

    newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{verification_remider}
"""
    if(verbose):
        print(">>>>>>> Start verification.")
    p2 = build_request_payload(system_prompt=verification_system_prompt, 
        question_prompt=newst
        )
    
    if(verbose):
        print(">>>>>>> Verification prompt:")
        print(json.dumps(p2, indent=4))

    if stop_event is not None and stop_event.is_set():
        if verbose:
            print("[verify_solution] Interrupted before verifier request dispatch.")
        return "", "no"

    res = send_api_request(get_api_key(), p2)
    out = extract_text_from_response(res) 

    if(verbose):
        print(">>>>>>> Verification results:")
        print(json.dumps(out, indent=4))

    check_correctness = (
        "Read the following verification report carefully, then decide whether the solution "
        "is correct (i.e., does NOT contain any critical error or major justification gap).\n\n"
        "--- VERIFICATION REPORT START ---\n"
        + out +
        "\n--- VERIFICATION REPORT END ---\n\n"
        "Based solely on the report above, output your conclusion in EXACTLY the following format "
        "(one line, nothing else after it):\n"
        "FINAL_ANSWER: yes\n"
        "or\n"
        "FINAL_ANSWER: no\n\n"
        "Use 'yes' only if the report concludes the solution is fully correct with no critical errors "
        "or major justification gaps. Use 'no' otherwise.\n"
        "You MUST end your response with the line 'FINAL_ANSWER: yes' or 'FINAL_ANSWER: no'."
    )
    prompt = build_request_payload(system_prompt="", question_prompt=check_correctness)

    if stop_event is not None and stop_event.is_set():
        if verbose:
            print("[verify_solution] Interrupted before verdict request.")
        return "", "no"

    r = send_api_request(get_api_key(), prompt)
    o = extract_text_from_response(r)

    if(verbose):
        print(">>>>>>> Is verification good?")
        print(json.dumps(o, indent=4))

    verdict = parse_verdict(o)
    if(verbose):
        print(f">>>>>>> Parsed verdict: {verdict}")

    bug_report = ""

    if verdict != "yes":
        bug_report = extract_bug_report(out)

    if(verbose):
        print(">>>>>>>Bug report:")
        print(json.dumps(bug_report, indent=4))

    return bug_report, verdict

def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
    """

    p1 = build_request_payload(system_prompt="",    question_prompt=check_complete_prompt)
    r = send_api_request(get_api_key(), p1)
    o = extract_text_from_response(r)

    print(o)
    return "yes" in o.lower()


def init_explorations(problem_statement, verbose=True, other_prompts=[], stop_event=None):
    if stop_event is not None and stop_event.is_set():
        print(">>>>>>> init_explorations skipped by early-stop signal.")
        return None, None, "", "no"

    p1  = build_request_payload(
            system_prompt=step1_prompt,
            question_prompt=problem_statement,
            #other_prompts=["* Please explore all methods for solving the problem, including casework, induction, contradiction, and analytic geometry, if applicable."]
            #other_prompts = ["You may use analytic geometry to solve the problem."]
            other_prompts = other_prompts
        )

    print(f">>>>>> Initial prompt.")
    print(json.dumps(p1, indent=4))

    if stop_event is not None and stop_event.is_set():
        print(">>>>>>> init_explorations interrupted before first request.")
        return None, None, "", "no"

    response1 = send_api_request(get_api_key(), p1)
    output1 = extract_text_from_response(response1)

    print(f">>>>>>> First solution: ") 
    print(json.dumps(output1, indent=4))

    print(f">>>>>>> Self improvement start:")
    # Append assistant reply and follow-up user turn to messages list for multi-turn
    messages = p1['messages'].copy()
    messages.append({"role": "assistant", "content": output1})
    messages.append({"role": "user", "content": self_improvement_prompt})
    p1 = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }

    if stop_event is not None and stop_event.is_set():
        print(">>>>>>> init_explorations interrupted before self-improvement request.")
        return None, None, "", "no"

    response2 = send_api_request(get_api_key(), p1)
    solution = extract_text_from_response(response2)
    print(f">>>>>>> Corrected solution: ")
    print(json.dumps(solution, indent=4))
    
    #print(f">>>>>>> Check if solution is complete:"  )
    #is_complete = check_if_solution_claimed_complete(output1)
    #if not is_complete:
    #    print(f">>>>>>> Solution is not complete. Failed.")
    #    return None, None, None, None
    
    print(f">>>>>>> Vefify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose, stop_event=stop_event)

    print(f">>>>>>> Initial verification: ")
    print(json.dumps(verify, indent=4))
    print(f">>>>>>> verify results: {good_verify}")
    
    return p1, solution, verify, good_verify

def run_solution_rollout(problem_statement, other_prompts=[], stop_event=None, run_idx=None):
    run_tag = f"[run {run_idx}] " if run_idx is not None else ""

    if stop_event is not None and stop_event.is_set():
        print(f"{run_tag}>>>>>>> Skip rollout because early-stop signal is already set.")
        return None, 0, False

    p1, solution, verify, good_verify = init_explorations(
        problem_statement,
        True,
        other_prompts,
        stop_event=stop_event,
    )

    if solution is None:
        print(f"{run_tag}>>>>>>> Failed in finding a complete solution.")
        return None, 0, False

    best_solution = solution
    best_correct_count = 1
    error_count = 0
    correct_count = 1
    success = False
    for i in range(MAX_EXPLORATION_ROUNDS):
        if stop_event is not None and stop_event.is_set():
            print(f"{run_tag}>>>>>>> Stop signal received, terminating this run early.")
            return best_solution, best_correct_count, False

        print(f"Number of iterations: {i}, number of corrects: {correct_count}, number of errors: {error_count}")

        try:
            if("yes" not in good_verify.lower()):
                # clear
                correct_count = 0
                error_count += 1

                #self improvement
                print(">>>>>>> Verification does not pass, correcting ...")
                # establish a new prompt that contains the solution and the verification

                base = build_request_payload(
                    system_prompt=step1_prompt,
                    question_prompt=problem_statement,
                    other_prompts=other_prompts
                )

                # Append previous solution and bug-report correction turn to messages list
                messages = base['messages'].copy()
                messages.append({"role": "assistant", "content": solution})
                correction_msg = f"{correction_prompt}\n\n{verify}"
                if is_parse_failure_report(verify):
                    correction_msg += f"\n\n{parse_reformat_prompt}"
                messages.append({"role": "user", "content": correction_msg})
                p1 = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                }

                print(">>>>>>> New prompt:")
                print(json.dumps(p1, indent=4))
                if stop_event is not None and stop_event.is_set():
                    print(f"{run_tag}>>>>>>> Stop signal received before correction request.")
                    return best_solution, best_correct_count, False
                response2 = send_api_request(get_api_key(), p1)
                solution = extract_text_from_response(response2)

                print(">>>>>>> Corrected solution:")
                print(json.dumps(solution, indent=4))

            print(f">>>>>>> Verify the solution.")
            if stop_event is not None and stop_event.is_set():
                print(f"{run_tag}>>>>>>> Stop signal received before verification.")
                return best_solution, best_correct_count, False

            verify, good_verify = verify_solution(problem_statement, solution, stop_event=stop_event)

            if("yes" in good_verify.lower()):
                print(">>>>>>> Solution is good, verifying again ...")
                correct_count += 1
                error_count = 0
                if correct_count >= best_correct_count:
                    best_solution = solution
                    best_correct_count = correct_count

            if(correct_count >= MAX_VERIFICATION_TRUE_ROUNDS):
                print(">>>>>>> Correct solution found.")
                print(json.dumps(solution, indent=4))
                if stop_event is not None:
                    stop_event.set()
                return solution, best_correct_count, True

            elif(error_count >= MAX_VERIFICATION_FALSE_ROUNDS):
                print(">>>>>>> Failed in finding a correct solution.")
                return best_solution, best_correct_count, False

        except Exception as e:
            print("Unexpected error:", e, "retry...")
            good_verify = "no"

    if(not success):
        print(">>>>>>> Failed in finding a correct solution.")
        return best_solution, best_correct_count, False

def format_final_output(problem_statement, solution, general_prompt_path, instruct_txt_path, out_path):
    """
    Format the final solution with the dataset prompt and problem instruction.
    """
    print(f">>>>>>> Format final output step.")

    try:
        with open(general_prompt_path, 'r', encoding='utf-8') as f:
            general_prompt_content = f.read().strip().replace('\\n', '\n')
    except Exception as e:
        print(f"Warning: cannot read general_prompt.txt ({e}), using empty string.")
        general_prompt_content = ""

    try:
        with open(instruct_txt_path, 'r', encoding='utf-8') as f:
            instruct_content = f.read().strip()
    except Exception as e:
        print(f"Warning: cannot read instruct file ({e}), using empty string.")
        instruct_content = ""

    detailed = extract_detailed_solution(solution)
    if not detailed:
        print("Warning: 'Detailed Solution' marker not found, using full solution.")
        detailed = solution.strip()

    clean_statement = re.sub(r'\*{3}.*?\*{3}\s*', '', problem_statement, flags=re.DOTALL).strip()

    format_prompt = (
        f"{general_prompt_content}\n\n"
        f"Problem: {clean_statement}\n\n"
        f"The following is a detailed mathematical solution to the above problem:\n"
        f"{detailed}\n\n"
        f"{instruct_content}"
    )

    print(">>>>>>> Format prompt:")
    print(format_prompt)

    payload = build_request_payload(system_prompt="", question_prompt=format_prompt)
    response = send_api_request(get_api_key(), payload)
    formatted_output = extract_text_from_response(response)

    print(">>>>>>> Formatted output:")
    print(json.dumps(formatted_output, indent=4))

    try:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        print(f">>>>>>> Formatted output saved to: {out_path}")
    except Exception as e:
        print(f"Error writing output file {out_path}: {e}")

    return formatted_output


def _detect_dataset_name(problem_file_path, dataset_name_arg=None):
    """Use the CLI dataset name, or infer it from a problems/<dataset>/ path."""
    if dataset_name_arg:
        return dataset_name_arg.strip().lower()

    abs_path = os.path.abspath(problem_file_path)
    parts = [p.lower() for p in abs_path.split(os.sep) if p]
    for idx, part in enumerate(parts):
        if part == "problems" and idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def _is_proof_dataset(problem_file_path, dataset_name_arg=None):
    dataset_name = _detect_dataset_name(problem_file_path, dataset_name_arg)
    return dataset_name in PROOF_DATASETS


def write_raw_solution(solution, out_path):
    """Write the generated solution directly to the output file."""
    print(">>>>>>> Skip formatting for proof dataset, write raw solution directly.")
    try:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(solution.strip() + "\n")
        print(f">>>>>>> Raw solution saved to: {out_path}")
    except Exception as e:
        print(f"Error writing output file {out_path}: {e}")


def _run_single_roll(run_idx, problem_statement, other_prompts, stop_event=None):
    """
    Run one independent rollout and capture exceptions as return values.
    """
    if stop_event is not None and stop_event.is_set():
        return {
            "run_idx": run_idx,
            "solution": None,
            "sol_count": 0,
            "ok": False,
            "error": None,
            "stopped": True,
        }

    print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>> Run {run_idx} started ...")
    try:
        sol, sol_count, ok = run_solution_rollout(
            problem_statement,
            other_prompts,
            stop_event=stop_event,
            run_idx=run_idx,
        )
        return {
            "run_idx": run_idx,
            "solution": sol,
            "sol_count": sol_count,
            "ok": ok,
            "error": None,
            "stopped": bool(stop_event.is_set() and not ok) if stop_event is not None else False,
        }
    except Exception as e:
        return {
            "run_idx": run_idx,
            "solution": None,
            "sol_count": 0,
            "ok": False,
            "error": str(e),
            "stopped": False,
        }


def _is_better_roll(candidate, current):
    """
    Compare two roll results.
    Priority: success > failure, then higher verification count, then smaller run_idx.
    """
    if current is None:
        return True
    cand_ok = bool(candidate.get("ok"))
    curr_ok = bool(current.get("ok"))
    if cand_ok != curr_ok:
        return cand_ok
    cand_count = int(candidate.get("sol_count", 0))
    curr_count = int(current.get("sol_count", 0))
    if cand_count != curr_count:
        return cand_count > curr_count
    return int(candidate.get("run_idx", 10**9)) < int(current.get("run_idx", 10**9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTS decode for math problems.')
    parser.add_argument('problem_file', nargs='?', default='problem_statement.txt')
    parser.add_argument('--log', '-l', type=str)
    parser.add_argument('--other_prompts', '-o', type=str)
    parser.add_argument('--max_runs', '-m', type=int, default=10)
    parser.add_argument('--parallel_runs', type=int, default=3)
    parser.add_argument('--out', type=str, help='Path to formatted output file (default: log_dir)')
    parser.add_argument('--dataset_name', type=str, help='Optional dataset name (e.g., imo24, imo25, proofbench)')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=DECODE_DRY_RUN,
        help='Run deterministic mock model responses without calling API_URL.',
    )

    args = parser.parse_args()
    DECODE_DRY_RUN = args.dry_run
    max_runs = args.max_runs
    parallel_runs = args.parallel_runs

    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(',')

    print(">>>>>>> Other prompts:")
    print(other_prompts)

    if args.log:
        if not set_log_file(args.log):
            sys.exit(1)
        print(f"Logging to file: {args.log}")
    if args.dry_run:
        print("[config] dry_run=1, API requests are skipped.")

    problem_dir = os.path.dirname(os.path.abspath(args.problem_file))
    problem_basename = os.path.basename(args.problem_file)          # e.g. amobench-1.txt
    problem_stem = os.path.splitext(problem_basename)[0]            # e.g. amobench-1

    general_prompt_path = os.path.join(problem_dir, "general_prompt.txt")
    instruct_txt_path = os.path.join(problem_dir, f"{problem_stem}_instruct.txt")

    if args.out:
        out_path = args.out
    else:
        if args.log:
            log_dir = os.path.dirname(os.path.abspath(args.log))
        else:
            log_dir = problem_dir

        out_base_dir = os.path.join(log_dir, "out")
        os.makedirs(out_base_dir, exist_ok=True)

        out_path = os.path.join(out_base_dir, f"{problem_stem}_out.txt")

    problem_statement = read_file_content(args.problem_file)
    final_solution = None
    final_best_count = 0
    success = False

    if max_runs <= 0:
        print(f">>>>>>> Invalid max_runs={max_runs}, reset to 1.")
        max_runs = 1

    if parallel_runs <= 0:
        print(f">>>>>>> Invalid parallel_runs={parallel_runs}, reset to 1.")
        parallel_runs = 1
    if parallel_runs > max_runs:
        print(f">>>>>>> parallel_runs={parallel_runs} exceeds max_runs={max_runs}, clamp to {max_runs}.")
        parallel_runs = max_runs

    print(f">>>>>>> Rolling {max_runs} runs with parallel_runs={parallel_runs} ...")

    best_roll = None
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
        futures = [
            executor.submit(_run_single_roll, i, problem_statement, other_prompts, stop_event)
            for i in range(max_runs)
        ]
        for future in as_completed(futures):
            result = future.result()
            run_idx = result["run_idx"]

            if result.get("stopped"):
                print(f">>>>>>> Run {run_idx} stopped by early-stop signal.")
                continue

            if result["error"] is not None:
                print(f">>>>>>> Error in run {run_idx}: {result['error']}")
                continue

            sol = result["solution"]
            sol_count = result["sol_count"]
            ok = result["ok"]

            if sol is not None and _is_better_roll(result, best_roll):
                best_roll = result
                print(
                    f">>>>>>> Updated best roll from run {run_idx} "
                    f"(verification count: {sol_count}, ok={ok})."
                )

            if ok:
                print(f">>>>>>> Found a correct solution in run {run_idx}.")
                stop_event.set()
                best_roll = result
                success = True
                cancelled = 0
                for pending in futures:
                    if pending is not future and pending.cancel():
                        cancelled += 1
                print(f">>>>>>> Early-stop signal sent. Cancelled {cancelled} pending runs.")
                break

    if best_roll is not None:
        final_solution = best_roll["solution"]
        final_best_count = best_roll["sol_count"]
        if not success:
            success = bool(best_roll["ok"])

    is_proof_dataset = _is_proof_dataset(args.problem_file, args.dataset_name)

    if final_solution is not None:
        status = "success" if success else "failed (using last available solution)"
        if is_proof_dataset:
            print(f">>>>>>> Decode status: {status}. Proof dataset detected, skip formatting.")
            write_raw_solution(solution=final_solution, out_path=out_path)
        else:
            print(f">>>>>>> Decode status: {status}. Proceeding to format output.")
            format_final_output(
                problem_statement=problem_statement,
                solution=final_solution,
                general_prompt_path=general_prompt_path,
                instruct_txt_path=instruct_txt_path,
                out_path=out_path,
            )
    else:
        print(">>>>>>> No solution available at all, skipping format output.")

    close_log_file()
