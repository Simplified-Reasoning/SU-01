import os
import json
import time
import argparse
import re
import base64
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

IMO_TEMPLATE = '''You are an expert grader for the International Mathematics Olympiad (IMO). Your task is to evaluate a proposed solution strictly and rigorously. Keep in mind the standards at the IMO are extremely high: only arguments that are logically sound, complete, and precise should be rewarded.

### General Scoring Rubric

Scores are assigned on a 0-7 scale. The general guidelines are:

* **7 Points (Correct):** The solution is complete, correct, and fully rigorous. If the submission contains incorrect attempts or lines of reasoning but ultimately presents a complete and correct solution, it should still be awarded full points; the presence of earlier, discarded work does not detract from the final correct proof.
* **6 Points (Almost Correct):** The solution is almost correct with a sound core argument, but contains minor errors in calculation or small gaps in logic. Missing proofs for major components, unjustified claims, or sketchy arguments are **not** eligible for 6 points.
* **1 Point (Partial Progress):** The solution demonstrates substantial progress explicitly mentioned in the grading guidelines. Initial observations, reformulating the problem without making substantive headway, or proving partial results not mentioned in the grading guidelines are generally **not** eligible for this score.
* **0 Points (Incorrect):** The solution doesn't make substantial progress that is a key step in the full solution or is fundamentally flawed. All partial progress without key results or lacking rigor also fall in this category.

### Input Data and Interpretation

You are provided with the following:

1. **Problem Statement:** The IMO problem.
2. **Ground Truth Solution:** A reference solution. Assume this solution is correct. It demonstrates one valid approach.
3. **Specific Grading Guidelines:** Criteria for awarding credit for this specific problem. These guidelines take precedence over the General Scoring Rubric, especially for partial credit.
4. **Proposed Solution:** The student submission.

### Evaluation Process

You must follow this structured process:

1. **Analyze References:** Meticulously read and understand the problem and Ground Truth Solution check the Specific Grading Guidelines. Identify the key steps for a complete solution and the criteria for partial credit.
2. **Step-by-Step Verification:** Verify the logical validity and rigor of every step. Identify all flaws, gaps, assumptions, and errors. **Make sure you fully understand every piece of logic behind each step of the proposed solution, you must be careful for solutions that 'pretend' to be correct.**
3. **Assess Progress:** Determine the extent of non-trivial progress made.
4. **Score Determination:** Compare the findings against the Specific Grading Guidelines and the General Rubric to determine the final score.

### Output Requirements

You must provide your final score in the format `<points>N out of 7</points>`. Ensure the `<points>` block is used **only once**, as your answer will be parsed based on the first `<points> </points>` block that appears in your whole response.

---

**PROBLEM STATEMENT**  
`{problem}`

**GROUND-TRUTH SOLUTION**  
`{solution}`

**SPECIFIC GRADING GUIDELINES**  
`{guidelines}`

**PROPOSED SOLUTION**  
`{student_answer}`

---

Present your detailed thought process and formal justification based on the scoring rubric and grading guidelines, and finally present your final score in the format below.

[Select one of the following options]

- `<points>7 out of 7</points>`
- `<points>6 out of 7</points>`
- `<points>1 out of 7</points>`
- `<points>0 out of 7</points>`
'''

# IMO_TEMPLATE = """
# You are an expert grader for the International Mathematics Olympiad (IMO). Your task is to evaluate a proposed solution strictly and rigorously. Keep in mind the standards at the IMO are extremely high: only arguments that are logically sound, complete, and precise should be rewarded. 

# ### General Scoring Rubric Scores are assigned on a 0-7 scale. The general guidelines are: 
# * **7 Points (Correct):** The solution is complete, correct, and fully rigorous. If the submission contains incorrect attempts or lines of reasoning but ultimately presents a complete and correct solution, it should still be awarded full points; the presence of earlier, discarded work does not detract from the final correct proof. 
# * **6 Points (Almost Correct):** The solution is almost correct with a sound core argument, but contains minor errors in calculation or small gaps in logic. Missing proofs for major components, unjustified claims, or sketchy arguments are **not** eligible for 6 points. 
# * **1 Point (Partial Progress):** The solution demonstrates substantial progress explicitly mentioned in the grading guidelines. Initial observations, reformulating the problem without making substantive headway, or proving partial results not mentioned in the grading guidelines are generally **not** eligible for this score. 
# * **0 Points (Incorrect):** The solution doesn’t make substantial progress that is a key step in the full solution or is fundamentally flawed. All partial progress without key results or lacking rigor also fall in this category.

# ### Input Data and Interpretation You are provided with the following: 
# 1. **Problem Statement:** The IMO problem. 
# 2. **Ground Truth Solution:** A reference solution. Assume this solution is correct. It demonstrates one valid approach. 
# 3. **Specific Grading Guidelines:** Criteria for awarding credit for this specific problem. These guidelines take precedence over the General Scoring Rubric, especially for partial credit. 
# 4. **Proposed Solution:** The student submission. 

# ### Evaluation Process You must follow this structured process: 
# 1.**Analyze References:** Meticulously read and understand the problem and Ground Truth Solution check the Specific Grading Guidelines. Identify the key steps for a complete solution and the criteria for partial credit. 
# 2.**Step-by-Step Verification:** Verify the logical validity and rigor of every step. Identify all flaws, gaps, assumptions, and errors. **Make sure you fully understand every piece of logic behind each step of the proposed solution, you must be careful for solutions that ’pretend’ to be correct.** 
# 3.**Assess Progress:** Determine the extent of non-trivial progress made.
# 4.**Score Determination:** Compare the findings against the Specific Grading Guidelines and the General Rubric to determine the final score. \

# ### Output Requirements You must provide your final score in the format N out of 7. Ensure the ‘‘ block is used **only once**, as your answer will be parsed based on the first <points> </points>
# block that appears in your whole response.

# **PROBLEM STATEMENT**
# {problem}
# **GROUND-TRUTH SOLUTION**
# {solution}
# **SPECIFIC GRADING GUIDELINES**
# {guidelines}
# **PROPOSED SOLUTION**
# {student_answer}

# Present your detailed thought process and formal justification
# based on the scoring rubric and grading guidelines, and finally
# present your final score in the format below.
# [Select one of the following options]
# <points>7 out of 7</points>
# <points>6 out of 7</points>
# <points>1 out of 7</points>
# <points>0 out of 7</points>
# """


NON_PROOF_TEMPLATE = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will
receive a question, an output sentence, and the correct answer. Your task is to determine if the output
sentence accurately answers the question based on the provided correct answer. Respond with either
[Correct] or [Incorrect].
-
Special considerations:
1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers
modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the
final answer is unclear or incorrect, respond with [Incorrect].
2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent such as 256/55=4.65,
respond with [Correct].
3. **Phycis Problems**: If the values match such as 3=3 \\, \\text{{GHz}} return [Correct].
4. **Explicit Options**: If the question provides explicit candidate answers, the output will be
considered correct if it clearly indicates the correct option's code or the correct option's content.
5. **No Explicit Options**: If the question does not provide explicit options, the output must align
with the correct answer in content and meaning to be considered [Correct].
-
Question: """{problem}"""
Output sentence: """{given_answer}"""
Correct answer: {ground_truth}
Judgement:
'''
# ----------------- CONFIG -----------------
MAX_TOKENS = 32768
RETRY_TIMES = 3
REQUEST_TIMEOUT = 1200
# ------------------------------------------

_PRINT_FIRST_PROMPT_LOCK = threading.Lock()
_PRINTED_FIRST_PROMPT = False
IMAGE_ROOT: Optional[Path] = None


def _extract_text_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content")
        if isinstance(content, str):
            parts.append(f"[{role}]\n{content}")
            continue
        if isinstance(content, list):
            text_chunks: List[str] = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    t = c.get("text", "")
                    if t:
                        text_chunks.append(t)
            if text_chunks:
                parts.append(f"[{role}]\n{''.join(text_chunks)}")
    return "\n\n".join(parts).strip()


def _count_images_in_messages(messages: List[Dict[str, Any]]) -> int:
    cnt = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image_url":
                    cnt += 1
    return cnt


def _get_problem_text_and_source(item: Dict[str, Any]) -> Tuple[str, str]:
    """统一解析 problem 文本及其来源字段"""
    question = item.get('question')
    if isinstance(question, str) and question.strip():
        return question, "question"

    prompt = item.get('prompt')
    if isinstance(prompt, str) and prompt.strip():
        return prompt, "prompt"

    return "", "empty"


def _as_prompt_text(value: Any) -> str:
    """Convert common JSON metadata values into readable prompt text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(_as_prompt_text(v) for v in value if _as_prompt_text(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _maybe_print_first_prompt(messages: List[Dict[str, Any]], idx: Optional[int] = None, max_chars: int = 12000):
    global _PRINTED_FIRST_PROMPT
    with _PRINT_FIRST_PROMPT_LOCK:
        if _PRINTED_FIRST_PROMPT:
            return
        _PRINTED_FIRST_PROMPT = True

    text_prompt = _extract_text_prompt_from_messages(messages)
    image_cnt = _count_images_in_messages(messages)

    prefix = f"[DEBUG] 第一个请求的 prompt" + (f"（索引 {idx}）" if idx is not None else "")
    print(prefix)
    print(f"[DEBUG] image_count={image_cnt}, text_chars={len(text_prompt)}")
    if len(text_prompt) > max_chars:
        print(text_prompt[:max_chars] + "\n[DEBUG] ... (truncated)")
    else:
        print(text_prompt)


def load_data(input_path: str) -> List[Dict[str, Any]]:
    """加载数据文件"""
    if input_path.endswith('.jsonl'):
        with open(input_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    elif input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError("不支持的文件格式")


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """将图像文件编码为base64字符串"""
    path = Path(image_path)
    image_root = IMAGE_ROOT or (Path(os.environ["MO_IMAGE_ROOT"]) if os.environ.get("MO_IMAGE_ROOT") else None)
    if not path.is_absolute() and image_root is not None:
        path = image_root / path

    try:
        if not path.exists():
            print(f"警告: 图像文件不存在: {path}")
            return None

        with path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"警告: 无法编码图像 {path}: {e}")
        return None


def create_messages(item: Dict[str, Any], information: str = None, text_only: bool = False) -> List[Dict[str, Any]]:
    """为每个数据项创建消息，根据is_proof区分proof和non-proof评测"""
    is_proof = item.get('is_proof', 'yes')
    problem_text, _ = _get_problem_text_and_source(item)
    # is_proof = 'no'
    student_answer = _as_prompt_text(item.get('response', '')).split(
        '</think>')[-1].strip().removesuffix('<|im_end|>')

    if is_proof == 'no':
        label = item.get('label', [])
        ground_truth = _as_prompt_text(label)
        formatted_prompt = NON_PROOF_TEMPLATE.format(
            problem=problem_text,
            given_answer=student_answer,
            ground_truth=ground_truth,
        )
    else:
        solution = item.get('label', '')
        guidelines = _as_prompt_text(
            item.get('rubrics')
            or item.get('grading_guidelines')
            or item.get('Grading guidelines')
            or 'No specific grading guidelines provided. Use the general rubric.'
        )
        formatted_prompt = IMO_TEMPLATE.format(
            problem=problem_text,
            solution=solution,
            guidelines=guidelines,
            student_answer=student_answer,
        )

    # 构建消息内容
    content = []
    if text_only:
        formatted_prompt = re.sub(r'\[figure(\d+)\]', '', formatted_prompt)
        content.append({"type": "text", "text": formatted_prompt})
    elif 'image_question' in item and item['image_question']:
        image_paths = item['image_question']
        if isinstance(image_paths, list):
            # 查找占位符模式 [figure1], [figure2] 等
            placeholder_pattern = r'\[figure(\d+)\]'
            placeholders = re.findall(placeholder_pattern, formatted_prompt)

            if placeholders and len(placeholders) == len(image_paths):
                # 根据占位符位置插入图片
                # 将文本按占位符分割，并插入对应的图片
                parts = re.split(placeholder_pattern, formatted_prompt)
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # 文本部分
                        if part.strip():
                            content.append({"type": "text", "text": part})
                    else:  # 占位符数字部分
                        fig_num = int(part)
                        if fig_num <= len(image_paths):
                            image_path = image_paths[fig_num - 1]  # 数组索引从0开始
                            base64_image = encode_image_to_base64(image_path)
                            if base64_image:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                })
            else:
                # 没找到占位符，把图片放在开头
                for image_path in image_paths:
                    base64_image = encode_image_to_base64(image_path)
                    if base64_image:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                # 然后添加文本
                content.append({"type": "text", "text": formatted_prompt})
    else:
        # 如果没有图片，直接添加文本
        content.append({"type": "text", "text": formatted_prompt})

    return [{"role": "user", "content": content}]


def build_verify_raw_input(item: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构建可落盘的 verify 原始输入信息，便于排查评测问题"""
    problem_text, problem_source = _get_problem_text_and_source(item)
    return {
        "problem_source": problem_source,
        "problem_text": problem_text,
        "text_prompt": _extract_text_prompt_from_messages(messages),
        "image_count": _count_images_in_messages(messages),
        "messages": messages,
    }


def request_with_retry(client: OpenAI, model: str, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Optional[str]]]:
    """带重试的API请求，使用OpenAI库流式处理"""
    for attempt in range(RETRY_TIMES):
        try:
            # 使用流式处理
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                # top_p=0.8,
                max_tokens=MAX_TOKENS,
                # extra_body={
                #     "repetition_penalty": 1.05,
                #     "chat_template_kwargs": {"enable_thinking": False}  # default to True
                # }
            )
            text_resp = response.choices[0]
            message = text_resp.message
            reasoning_content = getattr(message, "reasoning_content", None)
            return {'content': message.content, 'reasoning_content': reasoning_content}

        except Exception as e:
            print(f"[重试 {attempt+1}/{RETRY_TIMES}] 失败: {e}")
            if attempt < RETRY_TIMES - 1:
                time.sleep(1)
    return None


def process_single_item(item: Dict[str, Any], information: str, client: OpenAI, args) -> Dict[str, Any]:
    """处理单个数据项"""
    messages = create_messages(item, information, args.text_only)
    if getattr(args, "print_first_prompt", False):
        _maybe_print_first_prompt(messages)
    response = request_with_retry(
        client=client,
        model=args.model_name,
        messages=messages,
    )

    # 在原数据基础上添加test_result字段
    result_item = item.copy()
    result_item["verify_raw_input"] = build_verify_raw_input(item, messages)
    # 处理API失败的情况
    if response:
        result_item["reasoning_content"] = response['reasoning_content']
        result_item["prediction"] = response['content']
    else:
        result_item["reasoning_content"] = None
        result_item["prediction"] = None
        print("[警告] API请求失败，结果为None")

    return result_item


def thread_process_single_item(item: Dict[str, Any], information: str, args, idx: int, delay: float) -> Dict[str, Any]:
    """多线程处理单个数据项，带延迟，每个线程创建自己的client"""
    # 延迟发送请求
    time.sleep(delay)

    # 每个线程创建自己的 OpenAI client
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=REQUEST_TIMEOUT
    )

    messages = create_messages(item, information, args.text_only)
    if getattr(args, "print_first_prompt", False):
        _maybe_print_first_prompt(messages, idx=idx)
    print(f"[索引 {idx}] 发送请求")

    response = request_with_retry(
        client=client,
        model=args.model_name,
        messages=messages,
    )
    print(f"[索引 {idx}] 收到响应")

    # 在原数据基础上添加字段
    result_item = item.copy()
    result_item["verify_raw_input"] = build_verify_raw_input(item, messages)
    result_item["reasoning_content"] = response['reasoning_content'] if response else None
    result_item["prediction"] = response['content'] if response else None
    result_item["_data_index"] = idx

    return result_item


def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存结果到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def get_competition_name(file_path: str) -> str:
    """从文件路径提取竞赛名称"""
    return Path(file_path).stem


def should_use_proofbench(args, data_file: Path) -> (bool, str):
    """决定是否启用 proofbench 覆盖逻辑"""
    # 兼容旧参数：显式传了 --use_proofbench 则强制开启
    if getattr(args, "use_proofbench", False):
        return True, "显式参数 --use_proofbench"

    mode = getattr(args, "proofbench_mode", "auto")
    if mode == "on":
        return True, "proofbench_mode=on"
    if mode == "off":
        return False, "proofbench_mode=off"

    # auto 模式：按数据文件名自动识别
    file_name = data_file.name.lower()
    stem_name = data_file.stem.lower()
    if "proofbench" in file_name or "proofbench" in stem_name:
        return True, "auto: 数据文件名命中 proofbench"

    return False, "auto: 未命中 proofbench 关键字"


def run_concurrent(args):
    """多线程并发执行函数，每隔1秒发出一个请求，每个线程创建自己的client"""
    print("开始加载数据...")

    # 处理单个文件或目录
    data_files = []
    data_path = Path(args.data_path)

    if data_path.is_file():
        data_files = [data_path]
    elif data_path.is_dir():
        # 获取目录中的所有JSON文件
        data_files = list(data_path.glob('*.json')) + \
            list(data_path.glob('*.jsonl'))
    else:
        raise ValueError(f"数据路径不存在: {args.data_path}")

    # 处理每个数据文件
    for data_file in data_files:
        print(f"处理文件: {data_file}")

        # 加载数据
        data = load_data(str(data_file))

        use_proofbench, pb_reason = should_use_proofbench(args, data_file)
        print(f"proofbench覆盖判定: {use_proofbench}（{pb_reason}）")

        if use_proofbench:
            proofbench_path = args.proofbench_path
            if not proofbench_path:
                raise ValueError("已启用 proofbench 覆盖，但未提供 --proofbench_path")
            if not os.path.exists(proofbench_path):
                raise ValueError(f"proofbench 文件不存在: {proofbench_path}")

            print(f"!!!!!!使用 proofbench 的 rubric 数据: {proofbench_path}")
            ori_data = load_data(proofbench_path)
            if len(ori_data) < len(data):
                raise ValueError(
                    f"proofbench 数据量不足: proofbench={len(ori_data)}, eval_data={len(data)}")

            for i, d in enumerate(data):
                d['question'] = ori_data[i]['Problem']
                d['rubrics'] = ori_data[i]['Grading guidelines']
        else:
            print("使用原始数据（未启用 proofbench 覆盖）")
            proof_count = sum(1 for d in data if d.get('is_proof') == 'yes')
            non_proof_count = sum(1 for d in data if d.get('is_proof') == 'no')
            print(
                f"  Proof 题目: {proof_count}, Non-proof 题目: {non_proof_count}")
            for d in data:
                if 'question' not in d and 'prompt' in d:
                    d['question'] = d['prompt']

        print(f"找到 {len(data)} 条数据")
        if not data:
            print(f"警告: 文件 {data_file} 中没有可用数据，跳过。")
            continue

        competition_name = get_competition_name(str(data_file))
        model_name = args.model_name.split("/")[-1]

        information = data[0].get('information', '')

        iterable_data = data

        output_dir = os.path.join(
            args.output_dir, competition_name, model_name)
        suffix = f"-{args.run_suffix}" if args.run_suffix else ""
        output_file = f"{competition_name}-{model_name}{suffix}.json"
        output_path = os.path.join(output_dir, output_file)

        existing_results: List[Dict[str, Any]] = []
        processed_indices = set()

        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        # 断点续跑：仅把 prediction 非空的条目视为“已处理”
                        # 若 prediction 为 None/空串（通常是异常占位），则视为“未处理”，下次重试。
                        kept_results: List[Dict[str, Any]] = []
                        dropped_retryable = 0
                        for item_res in loaded:
                            idx = item_res.get("_data_index")
                            if not isinstance(idx, int):
                                kept_results.append(item_res)
                                continue

                            pred = item_res.get("prediction")
                            pred_text = pred.strip() if isinstance(pred, str) else None
                            if pred_text:
                                processed_indices.add(idx)
                                kept_results.append(item_res)
                            else:
                                dropped_retryable += 1

                        existing_results = kept_results
                        if dropped_retryable:
                            print(
                                f"检测到 {dropped_retryable} 条 prediction 为空的历史结果，将视为未处理并重试。",
                                flush=True,
                            )
                    else:
                        print(f"警告: {output_path} 内容格式不是列表，将覆盖写入。")
            except json.JSONDecodeError:
                print(f"警告: {output_path} 无法解析，将从头开始覆盖写入。")

        if processed_indices:
            print(f"检测到 {len(processed_indices)} 条已有有效结果，将跳过对应数据。")

        # 准备待处理的任务
        tasks_data = []
        delay_counter = 0

        for idx, item in enumerate(iterable_data, start=1):
            if idx in processed_indices:
                continue

            # 每个请求延迟 delay_counter 秒（以1秒为间隔递增）
            tasks_data.append(
                (item, information, args, idx, delay_counter * 1.0))
            delay_counter += 1

        print(f"将并发处理 {len(tasks_data)} 个请求，每个请求间隔1秒发出...")

        # 使用线程池执行所有任务
        results_list = []
        save_lock = threading.Lock()  # 用于保护文件写入的锁

        if tasks_data:
            # 使用 ThreadPoolExecutor，最大线程数为任务数（因为有延迟，不会同时发起）
            max_workers = min(len(tasks_data), 150)  # 限制最大线程数为50

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_data = {
                    # task_data[3] 是 idx
                    executor.submit(thread_process_single_item, *task_data): task_data[3]
                    for task_data in tasks_data
                }

                # 使用 tqdm 显示进度，使用 with 语句确保正确关闭
                with tqdm(total=len(future_to_data), desc="推理进度") as progress_bar:
                    # 等待任务完成
                    for future in as_completed(future_to_data):
                        try:
                            result = future.result()
                            results_list.append(result)

                            # 实时保存结果，避免程序崩溃导致数据丢失
                            # 按照 _data_index 排序以保持原始顺序
                            with save_lock:
                                all_results = existing_results + results_list
                                # 按 _data_index 排序
                                all_results_sorted = sorted(
                                    all_results, key=lambda x: x.get('_data_index', 0))
                                save_results(all_results_sorted, output_path)

                        except Exception as e:
                            idx = future_to_data[future]
                            print(f"[错误] 处理索引 {idx} 时发生异常: {e}")
                        finally:
                            progress_bar.update(1)

        print(f"结果已保存到: {output_path}")


def run_serial(args):
    """主要的串行执行函数"""
    print("开始加载数据...")

    # 初始化OpenAI客户端
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=REQUEST_TIMEOUT
    )

    # 处理单个文件或目录
    data_files = []
    data_path = Path(args.data_path)

    if data_path.is_file():
        data_files = [data_path]
    elif data_path.is_dir():
        # 获取目录中的所有JSON文件
        data_files = list(data_path.glob('*.json')) + \
            list(data_path.glob('*.jsonl'))
    else:
        raise ValueError(f"数据路径不存在: {args.data_path}")

    # 处理每个数据文件
    for data_file in data_files:
        print(f"处理文件: {data_file}")

        # 加载数据
        data = load_data(str(data_file))

        if any(d.get('is_proof') is not None for d in data):
            proof_count = sum(1 for d in data if d.get('is_proof') == 'yes')
            non_proof_count = sum(1 for d in data if d.get('is_proof') == 'no')
            print(
                f"  Proof 题目: {proof_count}, Non-proof 题目: {non_proof_count}")
            for d in data:
                if 'question' not in d and 'prompt' in d:
                    d['question'] = d['prompt']

        print(f"找到 {len(data)} 条数据")
        if not data:
            print(f"警告: 文件 {data_file} 中没有可用数据，跳过。")
            continue

        competition_name = get_competition_name(str(data_file))
        model_name = args.model_name.split("/")[-1]

        information = data[0].get('information', '')

        iterable_data = data

        output_dir = os.path.join(
            args.output_dir, competition_name, model_name)
        suffix = f"-{args.run_suffix}" if args.run_suffix else ""
        output_file = f"{competition_name}-{model_name}{suffix}.json"
        output_path = os.path.join(output_dir, output_file)

        existing_results: List[Dict[str, Any]] = []
        processed_indices = set()

        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        # 断点续跑：仅把 prediction 非空的条目视为“已处理”
                        kept_results: List[Dict[str, Any]] = []
                        dropped_retryable = 0
                        for item_res in loaded:
                            idx = item_res.get("_data_index")
                            if not isinstance(idx, int):
                                kept_results.append(item_res)
                                continue

                            pred = item_res.get("prediction")
                            pred_text = pred.strip() if isinstance(pred, str) else None
                            if pred_text:
                                processed_indices.add(idx)
                                kept_results.append(item_res)
                            else:
                                dropped_retryable += 1

                        existing_results = kept_results
                        if dropped_retryable:
                            print(
                                f"检测到 {dropped_retryable} 条 prediction 为空的历史结果，将视为未处理并重试。",
                                flush=True,
                            )
                    else:
                        print(f"警告: {output_path} 内容格式不是列表，将覆盖写入。")
            except json.JSONDecodeError:
                print(f"警告: {output_path} 无法解析，将从头开始覆盖写入。")

        if processed_indices:
            print(f"检测到 {len(processed_indices)} 条已有有效结果，将跳过对应数据。")

        total_items = len(iterable_data)
        initial_progress = min(len(processed_indices), total_items)
        progress_bar = tqdm(total=total_items, desc="推理进度",
                            initial=initial_progress)

        results = existing_results[:]

        for idx, item in enumerate(iterable_data, start=1):
            if idx in processed_indices:
                continue

            try:
                result = process_single_item(item, information, client, args)
                result["_data_index"] = idx
                results.append(result)
                save_results(results, output_path)
            except Exception as e:
                print(f"[错误] 处理索引 {idx} 时发生异常: {e}")
                # 即使失败也保存一个占位结果，避免跳过
                result = item.copy()
                result["_data_index"] = idx
                try:
                    fallback_messages = create_messages(item, information, args.text_only)
                    result["verify_raw_input"] = build_verify_raw_input(item, fallback_messages)
                except Exception:
                    # 构造 raw_input 也失败时，至少保留 problem 来源信息
                    problem_text, problem_source = _get_problem_text_and_source(item)
                    result["verify_raw_input"] = {
                        "problem_source": problem_source,
                        "problem_text": problem_text,
                        "text_prompt": None,
                        "image_count": 0,
                        "messages": None,
                    }
                result["reasoning_content"] = None
                result["prediction"] = None
                result["error"] = str(e)
                results.append(result)
                save_results(results, output_path)
            finally:
                progress_bar.update(1)

        progress_bar.close()
        print(f"结果已保存到: {output_path}")


# ----------------- ENTRY -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO proof judging script with serial and concurrent modes.")
    parser.add_argument("--data_path", "--data-path", type=str,
                        required=True, help="Eval input JSON/JSONL file or directory.")
    parser.add_argument("--output_dir", "--output-dir", type=str,
                        default="results", help="Output directory.")
    parser.add_argument("--api_key", "--api-key", type=str,
                        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_TOKEN"),
                        help="API key. Defaults to OPENAI_API_KEY or OPENAI_API_TOKEN.")
    parser.add_argument("--base_url", "--base-url", type=str,
                        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                        help="OpenAI-compatible API base URL.")
    parser.add_argument("--model_name", "--model-name", type=str,
                        default="gemini-2.5-pro", help="Judge model name.")
    parser.add_argument("--run_suffix", type=str,
                        default="", help="输出文件名后缀，用于区分多次运行")
    parser.add_argument("--text_only", action="store_true", help="是否只输出文本结果")
    parser.add_argument("--image_root", "--image-root", type=Path,
                        default=None, help="Optional root for relative image paths. Can also be set with MO_IMAGE_ROOT.")
    parser.add_argument("--print_first_prompt", action="store_true",
                        help="是否打印第一个实际发送给模型的 prompt（只打印一次）")
    parser.add_argument("--concurrent", action="store_true",
                        help="使用并发模式（每秒发送一个请求），默认为串行模式")
    parser.add_argument("--proofbench_mode", type=str, choices=["auto", "on", "off"],
                        default="off", help="proofbench覆盖策略：auto(按文件名自动识别)/on(强制开启)/off(强制关闭)")
    parser.add_argument("--use_proofbench", action="store_true",
                        help="兼容旧参数：等价于 proofbench_mode=on")
    parser.add_argument("--proofbench_path", type=str,
                        default=None,
                        help="proofbench 数据路径（在 proofbench_mode=on 或 auto命中时生效）")

    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("缺少 API key：请传 --api-key 或设置 OPENAI_API_KEY / OPENAI_API_TOKEN")
    IMAGE_ROOT = args.image_root
    os.makedirs(args.output_dir, exist_ok=True)

    execution_mode = "并发（1秒间隔）" if args.concurrent else "串行"

    print(f"配置信息:")
    print(f"  数据路径: {args.data_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  模型名称: {args.model_name}")
    print(f"  输出后缀: {args.run_suffix or '无'}")
    print(f"  是否只输出文本结果: {args.text_only}")
    print(f"  是否打印第一个prompt: {args.print_first_prompt}")
    print(f"  执行模式: {execution_mode}")
    effective_pb_mode = "on" if args.use_proofbench else args.proofbench_mode
    print(f"  proofbench覆盖模式: {effective_pb_mode}")
    if effective_pb_mode in ("on", "auto"):
        print(f"  proofbench路径: {args.proofbench_path}")

    if args.concurrent:
        # 使用并发模式（多线程，每个线程创建自己的client）
        run_concurrent(args)
    else:
        # 使用串行模式
        run_serial(args)
