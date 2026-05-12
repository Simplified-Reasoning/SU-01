from .math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from .p1 import solution2answer


def get_deepscaler_rule_based_reward(response, label):
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    else:
        return 0

    num_questions_to_answer = len(label)
    
    extracted_answers = solution2answer(str(model_solution), num_answers=num_questions_to_answer)
    ground_truths = [solution2answer(str(gt), return_origin=True)[0] for gt in label]
    # model_answer = extract_answer(model_solution)
    # if model_answer is None:
    #     return 0
    # if label == "":
    #     return 0

    # Convert single answer to list for uniform processing
    # assert isinstance(label, (str, float, int))
    # ground_truths = [label] if isinstance(label, str) else label

    # Process each ground truth
    # processed_ground_truths = []
    # for truth in ground_truths:
    #     truth = str(truth)
    #     if "\\boxed" in truth:
    #         processed_truth = extract_answer(truth)
    #         if processed_truth is not None:
    #             processed_ground_truths.append(processed_truth)
    #     else:
    #         processed_ground_truths.append(truth)

    # if not processed_ground_truths:
    #     return 0

    # Check against all possible correct answers
    # for ground_truth in processed_ground_truths:
    #     is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
    #     if is_correct:
    #         return 1
    score_list = []
    for extracted_pred, ground_truth in zip(extracted_answers, ground_truths):
        is_correct = grade_answer_mathd(extracted_pred, ground_truth) or grade_answer_sympy(extracted_pred, ground_truth)
        score_list.append(is_correct)

    return sum(score_list) / num_questions_to_answer
