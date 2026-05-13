#!/usr/bin/env python3
"""Direct decoding entry point for SU-01 evaluation inputs.

The script supports two modes:
- a single ``.txt`` problem file, writing one output file;
- a JSONL file, writing one output per item.

All service details are configured through CLI flags or environment variables.
Use ``--dry-run`` for smoke tests that should not contact a model server.
"""
import os
import json
import requests
import argparse
import re
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, local

# Per-problem logging is bound to the active worker thread.
_log_lock = Lock()
_original_print = print
_thread_local = local()

def _get_problem_log_file():
    """Return the log file handle bound to the current thread."""
    return getattr(_thread_local, "problem_log_file", None)

def log_print(msg: str) -> None:
    """Print to stdout and mirror to the current per-problem log."""
    with _log_lock:
        _original_print(msg)
        f = _get_problem_log_file()
        if f is not None:
            try:
                f.write(msg + '\n')
                f.flush()
            except Exception:
                pass

def set_problem_log_file(file_handle) -> None:
    """Bind a per-problem log file to the current thread."""
    _thread_local.problem_log_file = file_handle

def clear_problem_log_file() -> None:
    """Clear the current thread's per-problem log binding."""
    if hasattr(_thread_local, "problem_log_file"):
        del _thread_local.problem_log_file

def log_file_only(msg: str) -> None:
    """Write only to the current per-problem log."""
    f = _get_problem_log_file()
    if f is not None:
        with _log_lock:
            try:
                f.write(msg + '\n')
                f.flush()
            except Exception:
                pass

# --- CONFIGURATION ---
MODEL_NAME = os.environ.get("MODEL_NAME", "SU01")
API_URL = os.environ.get("API_URL", "http://localhost:34883/v1/chat/completions")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "160000"))
_TOP_K_RAW = os.environ.get("TOP_K")
TOP_K = int(_TOP_K_RAW) if _TOP_K_RAW not in (None, "") else -1
TOP_P = float(os.environ.get("TOP_P", "0.95"))
API_TIMEOUT = float(os.environ.get("API_TIMEOUT", "432000000"))
DEFAULT_NO_PROXY = "localhost,127.0.0.1"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
SKIP_INSTRUCTION = os.environ.get("SKIP_INSTRUCTION", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

QUESTION_FORMAT_SEP = "\nAfter solving the above problem,"


def configure_proxy_env() -> None:
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


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


configure_proxy_env()
DECODE_DRY_RUN = _env_truthy("DECODE_DRY_RUN")


class LocalServerTester:
    def __init__(
        self,
        api_url: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        max_workers: int = MAX_WORKERS,
        problems_root: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = self.get_api_key()
        self.max_workers = max_workers
        self.print_lock = Lock()
        self.dry_run = dry_run

        self.problems_root = problems_root
        self.general_prompt_content: Optional[str] = None
        self._instruct_cache: Dict[str, str] = {}

        if self.problems_root:
            general_prompt_path = os.path.join(self.problems_root, "general_prompt.txt")
            if os.path.isfile(general_prompt_path):
                try:
                    with open(general_prompt_path, "r", encoding="utf-8") as f:
                        self.general_prompt_content = f.read().strip().replace('\\n', '\n')
                except Exception:
                    self.general_prompt_content = None
        
    @staticmethod
    def get_api_key() -> str:
        """Return the API key used for OpenAI-compatible endpoints."""
        api_key = os.getenv("OPENAI_API_KEY", "dummy")
        return api_key
    
    def safe_print(self, msg: str) -> None:
        """Thread-safe print that also writes to the active log file."""
        log_print(msg)
    
    def strip_thinking(self, content: str) -> str:
        """Remove hidden reasoning markup when present.

<details type="reasoning" done="true" duration="0">
<summary>Thought for 0 seconds</summary>
&gt; ...
</details>
        """
        if '</think>' in content:
            return content.split('</think>', 1)[-1].lstrip('\n')
        return content

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Make a metadata value safe for use as a filename stem."""
        return name.replace("/", "_").replace("\\", "_").strip()

    def parse_question_and_instruction(self, question_text: str) -> tuple[str, str]:
        """
        Split metadata.question into the problem statement and output instruction.

        - pure_question: content before the separator.
        - instruct_content: content after the separator, including the marker.
        """
        question_text = (question_text or "").strip()
        if not question_text:
            return "", ""

        if QUESTION_FORMAT_SEP in question_text:
            before, after = question_text.split(QUESTION_FORMAT_SEP, 1)
            return before.strip(), (QUESTION_FORMAT_SEP + after).strip()

        # Handle inputs where the separator is missing its leading newline.
        if "After solving the above problem," in question_text:
            marker = "After solving the above problem,"
            idx = question_text.find(marker)
            if idx != -1:
                before = question_text[:idx].strip()
                after = question_text[idx + len(marker):]
                return before, (marker + after).strip()

        return question_text, ""

    def clean_statement(self, statement: str) -> str:
        """Remove a leading '*** Problem Statement ***' heading if present."""
        s = (statement or "").strip()
        return re.sub(r'\*{3}.*?\*{3}\s*', '', s, flags=re.DOTALL).strip()

    def get_instruct_from_files(self, stem: str) -> Optional[str]:
        """Read <problems_root>/<stem>_instruct.txt if it exists."""
        if not self.problems_root or not stem:
            return None
        if stem in self._instruct_cache:
            return self._instruct_cache[stem]

        instruct_path = os.path.join(self.problems_root, f"{stem}_instruct.txt")
        if not os.path.isfile(instruct_path):
            self._instruct_cache[stem] = ""
            return None

        try:
            with open(instruct_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                self._instruct_cache[stem] = content
                return content
        except Exception:
            self._instruct_cache[stem] = ""
            return None

    def get_stem_from_item(self, item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}
        stem = metadata.get("source_id") or metadata.get("index") or metadata.get("problem_idx") or metadata.get("problem_id") or "unknown"
        try:
            stem = str(stem)
        except Exception:
            stem = "unknown"
        return self.sanitize_filename(stem)

    def build_prompt_from_item(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Build a prompt from general_prompt, the problem statement, and instruction.

        The problem statement is parsed from metadata.question when possible.
        The instruction is read from <stem>_instruct.txt first, then falls back
        to the instruction embedded in metadata.question.
        """
        metadata = item.get("metadata") or {}
        question_text = metadata.get("question")

        prompt_list = item.get("prompt") or []
        if not question_text and prompt_list:
            question_text = (prompt_list[0] or {}).get("content", "")

        pure_question, instruct_from_meta = self.parse_question_and_instruction(question_text or "")

        # If parsing fails, fall back to the original prompt content.
        if not pure_question:
            if prompt_list:
                return (prompt_list[0] or {}).get("content", "")
            return None

        clean_statement = self.clean_statement(pure_question)
        if SKIP_INSTRUCTION:
            return f"Problem: {clean_statement}"

        # general_prompt.txt is preferred; otherwise fall back to prompt content.
        if not self.general_prompt_content:
            if prompt_list:
                return (prompt_list[0] or {}).get("content", "")
            return None

        stem = self.get_stem_from_item(item)
        instruct_from_files = self.get_instruct_from_files(stem) if self.problems_root else None

        if SKIP_INSTRUCTION:
            instruct_content = ""
        else:
            instruct_content = (
                instruct_from_files
                if instruct_from_files is not None
                else instruct_from_meta
            )
        instruct_content = instruct_content or ""

        # Add a newline when file-backed instruction text has been stripped.
        if instruct_content and not instruct_content.startswith("\n"):
            instruct_content = "\n" + instruct_content

        if not instruct_content:
            return f"{self.general_prompt_content}\n\nProblem: {clean_statement}"
        return f"{self.general_prompt_content}\n\nProblem: {clean_statement}{instruct_content}"
    
    def build_payload(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Build an OpenAI-compatible chat completion payload."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": TOP_P,
        }
        if TOP_K is not None:
            payload["top_k"] = TOP_K
        return payload
    
    def send_request(self, payload: Dict[str, Any]) -> tuple:
        """Send the API request and return (content, response_data)."""
        if self.dry_run:
            prompt_preview = ""
            try:
                messages = payload.get("messages") or []
                prompt_preview = (messages[-1].get("content") or "")[:200]
            except Exception:
                prompt_preview = ""
            content = (
                "[DRY_RUN] direct_gen generated this deterministic response.\n\n"
                f"Model: {self.model_name}\n"
                f"Prompt preview: {prompt_preview}"
            )
            response_data = {
                "dry_run": True,
                "choices": [{"message": {"content": content}}],
            }
            return content, response_data

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=API_TIMEOUT,
            )
            response.raise_for_status()
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            return self.strip_thinking(content), response_data
        except requests.exceptions.RequestException as e:
            self.safe_print(f"API request failed: {e}")
            return None, None
        except (KeyError, IndexError, TypeError) as e:
            self.safe_print(f"Failed to parse API response: {e}")
            return None, None
    
    def process_single_item(self, item: Dict[str, Any], output_dir: str) -> tuple:
        """Process one item and return (success, problem_idx, elapsed_time)."""
        try:
            metadata = item.get("metadata", {})
            problem_idx = metadata.get("problem_idx", "unknown")

            content = self.build_prompt_from_item(item)
            if not content:
                self.safe_print("Warning: build_prompt_from_item returned empty content; skipping.")
                return False, str(problem_idx), 0.0

            out_subdir = os.path.join(output_dir, "out")
            output_filename = f"{problem_idx}_out.txt"
            output_path = os.path.join(out_subdir, output_filename)

            if os.path.exists(output_path):
                self.safe_print(f"Problem #{problem_idx} already has output; skipping (resume).")
                return True, problem_idx, 0.0
            
            output_basename = os.path.basename(output_dir.rstrip(os.sep)) or "run"
            problem_log_path = os.path.join(output_dir, f"{output_basename}-{problem_idx}.log")
            problem_log_file = open(problem_log_path, 'w', encoding='utf-8')
            set_problem_log_file(problem_log_file)
            try:
                problem_log_file.write(f"Logging to file: {problem_log_path}\n\n")
                problem_log_file.write(f">>>>>>>>>>>>>>>>>>>>>>>>>> Processing problem #{problem_idx} ...\n")
                problem_log_file.flush()
                
                start_time = datetime.now()
                timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
                
                self.safe_print(f"\n{'='*80}")
                self.safe_print(f"[{timestamp_str}] Processing problem #{problem_idx}")
                self.safe_print(f"{'='*80}")
                
                self.safe_print(f"[{timestamp_str}] >>>>>> Initial prompt.")
                payload = self.build_payload(content)
                log_file_only(json.dumps(payload, indent=4, ensure_ascii=False))
                
                response_content, response_data = self.send_request(payload)
                end_time = datetime.now()
                end_ts = end_time.strftime('%Y-%m-%d %H:%M:%S')
                
                if response_content is None:
                    self.safe_print(f"[{end_ts}] >>>>>> Request failed")
                    self.safe_print(f"Problem #{problem_idx} failed.")
                    return False, problem_idx, 0.0
                
                self.safe_print(f"[{end_ts}] >>>>>> Response:")
                if response_data is not None:
                    log_file_only(json.dumps(response_data, indent=2, ensure_ascii=False))
                
                elapsed_time = (end_time - start_time).total_seconds()
                self.safe_print(f"Problem #{problem_idx} completed in {elapsed_time:.2f}s.")
                self.safe_print(f"Saved to: {output_path}")

                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(response_content)
                return True, problem_idx, elapsed_time
            finally:
                clear_problem_log_file()
                try:
                    problem_log_file.close()
                except Exception:
                    pass
            
        except Exception as e:
            self.safe_print(f"Error while processing item: {e}")
            import traceback
            self.safe_print(traceback.format_exc())
            metadata = item.get("metadata", {})
            problem_idx = metadata.get("problem_idx", "unknown")
            return False, problem_idx, 0.0
    
    def process_jsonl_file(self, input_path: str, output_dir: str, log_path: Optional[str] = None):
        """Process a JSONL file with one output and one log per item."""
        if not os.path.exists(input_path):
            self.safe_print(f"Error: input file does not exist: {input_path}")
            return

        output_basename = os.path.basename(output_dir.rstrip(os.sep)) or "run"
        self.safe_print(f"Per-problem logs: {os.path.join(output_dir, output_basename + '-<problem_idx>.log')}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.safe_print(f"Created output directory: {output_dir}")
        out_subdir = os.path.join(output_dir, "out")
        if not os.path.exists(out_subdir):
            os.makedirs(out_subdir, exist_ok=True)
            self.safe_print(f"Created output subdirectory: {out_subdir}")
        
        self.safe_print(f"\nProcessing file: {input_path}")
        self.safe_print(f"Output directory: {output_dir}")
        self.safe_print(f"Config: Model={self.model_name}, Temp={self.temperature}, MaxTokens={self.max_tokens}")
        self.safe_print(f"Max workers: {self.max_workers}")
        
        success_count = 0
        failed_items = []
        processing_details = []
        items = []
        
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append((line_num, item))
                except json.JSONDecodeError as e:
                    self.safe_print(f"Warning: JSON parse error on line {line_num}: {e}")
                    failed_items.append(f"line_{line_num}")
                    continue
        
        total_count = len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_meta = {
                executor.submit(self.process_single_item, item, output_dir): (line_num, item)
                for line_num, item in items
            }
            
            for future in as_completed(future_to_meta):
                line_num, item = future_to_meta[future]
                try:
                    success, problem_idx, elapsed_time = future.result()
                except Exception as e:
                    self.safe_print(f"Worker task raised an exception: {e}")
                    import traceback
                    self.safe_print(traceback.format_exc())
                    metadata = item.get("metadata", {})
                    problem_idx = metadata.get("problem_idx", f"line_{line_num}")
                    success = False
                    elapsed_time = 0.0
                
                processing_details.append({
                    'problem_idx': problem_idx,
                    'success': success,
                    'elapsed_time': elapsed_time
                })
                
                if success:
                    success_count += 1
                else:
                    failed_items.append(problem_idx)
        
        self.safe_print(f"\n{'='*80}")
        self.safe_print("Processing complete.")
        self.safe_print(f"   Total: {total_count}")
        self.safe_print(f"   Successful: {success_count}")
        self.safe_print(f"   Failed: {total_count - success_count}")
        self.safe_print(f"   Output directory: {output_dir}")
        
        if failed_items:
            self.safe_print(f"\nFailed item ids: {', '.join(map(str, failed_items))}")
        
        self.safe_print(f"{'='*80}\n")
        
        summary_path = os.path.join(output_dir, "_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Processing Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {input_path}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Output files (per problem): {os.path.join(output_dir, 'out', '<problem_idx>_out.txt')}\n")
            f.write(f"Log files (per problem): {os.path.join(output_dir, output_basename + '-<problem_idx>.log')}\n")
            f.write(f"API URL: {self.api_url}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Temperature: {self.temperature}\n")
            f.write(f"Max Tokens: {self.max_tokens}\n")
            f.write(f"Max Workers: {self.max_workers}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"Total items: {total_count}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {total_count - success_count}\n")
            
            if failed_items:
                f.write(f"\nFailed items: {', '.join(map(str, failed_items))}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Processing Details:\n")
            f.write(f"{'='*80}\n")
            for detail in processing_details:
                status = "✓" if detail['success'] else "✗"
                f.write(f"{status} Problem {detail['problem_idx']}: {detail['elapsed_time']:.2f}s\n")
            
            f.write(f"{'='*80}\n")
            
            successful_times = [d['elapsed_time'] for d in processing_details if d['success']]
            if successful_times:
                avg_time = sum(successful_times) / len(successful_times)
                f.write(f"\nAverage processing time: {avg_time:.2f}s\n")
                f.write(f"Total processing time: {sum(successful_times):.2f}s\n")
        
        self.safe_print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Direct decode for a single txt problem or a JSONL input file.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input .txt problem file or JSONL file.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory. Default: <input_file>_output/',
        default=None
    )
    parser.add_argument(
        '--api-url',
        type=str,
        help=f'API URL. Default: {API_URL}',
        default=API_URL
    )
    parser.add_argument(
        '--model',
        type=str,
        help=f'Model name. Default: {MODEL_NAME}',
        default=MODEL_NAME
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help=f'Temperature. Default: {TEMPERATURE}',
        default=TEMPERATURE
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        help=f'Maximum output tokens. Default: {MAX_TOKENS}',
        default=MAX_TOKENS
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        help=f'Maximum worker threads. Default: {MAX_WORKERS}',
        default=MAX_WORKERS
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=None,
        help='Deprecated; per-problem logs are saved under the output directory.'
    )

    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Single-problem output path. Default: <log_dir>/out/<stem>_out.txt.',
    )

    parser.add_argument(
        '--problems-root',
        type=str,
        default=None,
        help='Problems root containing general_prompt.txt and *_instruct.txt.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=DECODE_DRY_RUN,
        help='Build prompts and write deterministic outputs without calling API_URL.',
    )
    
    args = parser.parse_args()

    input_abs = os.path.abspath(args.input_file)
    if input_abs.lower().endswith(".txt"):
        problem_dir = os.path.dirname(input_abs)
        problem_stem = os.path.splitext(os.path.basename(input_abs))[0]

        problems_root = problem_dir
        tester = LocalServerTester(
            api_url=args.api_url,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
            problems_root=problems_root,
            dry_run=args.dry_run,
        )

        def _read_text(path: str) -> str:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return ""

        general_prompt_path = os.path.join(problem_dir, "general_prompt.txt")
        instruct_txt_path = os.path.join(problem_dir, f"{problem_stem}_instruct.txt")

        general_prompt_content = "" if SKIP_INSTRUCTION else _read_text(general_prompt_path).strip()
        general_prompt_content = general_prompt_content.replace("\\n", "\n")

        instruct_content = "" if SKIP_INSTRUCTION else _read_text(instruct_txt_path).strip()

        problem_statement = _read_text(input_abs)
        clean_statement = tester.clean_statement(problem_statement)

        if SKIP_INSTRUCTION:
            prompt = f"Problem: {clean_statement}".strip()
        else:
            prompt = f"{general_prompt_content}\n\nProblem: {clean_statement}\n\n{instruct_content}".strip()

        payload = tester.build_payload(prompt)
        log_prefix = f"[single] {problem_stem}"

        log_file_handle = None
        try:
            if args.log:
                os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
                log_file_handle = open(args.log, "w", encoding="utf-8")
                set_problem_log_file(log_file_handle)
                log_file_handle.write(f"Logging to file: {args.log}\n\n")
                log_file_handle.write(f">>>>>>>>>>>>>>>>>>>>>>>>>> Processing problem #{problem_stem} ...\n")
                log_file_handle.flush()

            start_time = datetime.now()
            start_ts = start_time.strftime("%Y-%m-%d %H:%M:%S")
            tester.safe_print(f"\n{'='*80}")
            tester.safe_print(f"📝 [{start_ts}] {log_prefix} Start")
            tester.safe_print(f"{'='*80}")
            tester.safe_print(f"[{start_ts}] >>>>>> Initial prompt.")
            if SKIP_INSTRUCTION:
                tester.safe_print("[config] SKIP_INSTRUCTION=1, using problem-only prompt.")
            if args.dry_run:
                tester.safe_print("[config] dry_run=1, API request is skipped.")
            log_file_only(json.dumps(payload, indent=4, ensure_ascii=False))

            response_content, response_data = tester.send_request(payload)
            if response_content is None:
                tester.safe_print(f"{log_prefix} request failed.")
                sys.exit(1)

            end_time = datetime.now()
            end_ts = end_time.strftime("%Y-%m-%d %H:%M:%S")
            tester.safe_print(f"[{end_ts}] >>>>>> Response:")
            if response_data is not None:
                log_file_only(json.dumps(response_data, indent=2, ensure_ascii=False))

            elapsed_time = (end_time - start_time).total_seconds()

        except Exception as e:
            try:
                tester.safe_print(f"{log_prefix} ERROR: {e}")
            except Exception:
                print(f"{log_prefix} ERROR: {e}")
            sys.exit(1)
        finally:
            clear_problem_log_file()
            try:
                if log_file_handle is not None:
                    log_file_handle.close()
            except Exception:
                pass

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

        out_parent = os.path.dirname(os.path.abspath(out_path))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response_content)

        tester.safe_print(f"{log_prefix} Done ({elapsed_time:.2f}s)")
        tester.safe_print(f"Saved to: {out_path}")
        return
    
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        input_dir = os.path.dirname(os.path.abspath(args.input_file))
        args.output = os.path.join(input_dir, f"{base_name}_output")

    problems_root = args.problems_root
    if not problems_root:
        input_abs = os.path.abspath(args.input_file)
        input_dir = os.path.dirname(input_abs)
        dataset_guess = os.path.splitext(os.path.basename(input_abs))[0]
        if dataset_guess.endswith("_type"):
            dataset_guess = dataset_guess[: -len("_type")]

        candidate_roots = []
        candidate_roots.append(os.path.join(input_dir, "problems", dataset_guess))
        candidate_roots.append(os.path.join(os.path.dirname(input_dir), "problems", dataset_guess))

        for cand in candidate_roots:
            if os.path.isfile(os.path.join(cand, "general_prompt.txt")):
                problems_root = cand
                break

    tester = LocalServerTester(
        api_url=args.api_url,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        problems_root=problems_root,
        dry_run=args.dry_run,
    )
    
    tester.process_jsonl_file(args.input_file, args.output, log_path=args.log if args.log else None)


if __name__ == "__main__":
    main()
