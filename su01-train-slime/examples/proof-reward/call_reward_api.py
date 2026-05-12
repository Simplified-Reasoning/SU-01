"""
Send a test request to the reward server.

Usage:
    python call_reward_api.py --proof            # proof verification
    python call_reward_api.py                    # normal (non-proof) scoring
"""
import argparse
import json
import os

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://10.102.247.21", help="Reward server host (with scheme)")
    parser.add_argument("--port", type=int, default=8021, help="Reward server port")
    parser.add_argument("--proof", action="store_true", help="Use proof verification payload")
    parser.add_argument("--reviews", type=int, default=3, help="Reviews count when using pessimistic proof verifier")
    return parser.parse_args()


def build_payload(use_proof: bool, reviews: int):
    if use_proof:
        return {
            "response": """"
**Solution:**

Let $A_n$ denote the number of perfect powers less than or equal to $n$. A perfect power is an integer of the form $a^b$ where $a \ge 1, b \ge 2$ are integers. We want to prove that there are infinitely many positive integers $n$ such that $A_n$ divides $n+2024$.

Let $P=\{p_1, p_2, p_3, \dots\}$ be the set of perfect powers in increasing order. So $p_1=1, p_2=4, p_3=8, p_4=9, \dots$.
By definition, $A_n=k$ if and only if $p_k \le n < p_{k+1}$.
We are looking for $n$ such that $A_n=k$ and $k \mid n+2024$ for some $k$.
For a fixed $k$, $A_n=k$ restricts $n$ to the interval $I_k=[p_k, p_{k+1}-1]$. We need to know if there exists $n \in I_k$ such that $n \equiv -2024 \pmod k$.

The interval $I_k$ contains $p_{k+1}-p_k$ consecutive integers. If the length $p_{k+1}-p_k \ge k$, then $I_k$ contains at least $k$ consecutive integers. In this case, $I_k$ must contain an integer for every residue class modulo $k$. In particular, it must contain an integer $n_k$ such that $n_k \equiv -2024 \pmod k$.
For this $n_k$, we have $A_{n_k}=k$ since $n_k \in I_k$. Thus $A_{n_k} \mid n_k+2024$.

Let $S=\{k \in \mathbb{Z}^+ \mid p_{k+1}-p_k \ge k\}$. If $S$ is an infinite set, then for each $k \in S$, we can find such an $n_k$. The values $n_k$ for $k \in S$ are distinct because if $k_1 < k_2$ are in $S$, then $n_{k_1} \in I_{k_1}$ and $n_{k_2} \in I_{k_2}$. Since $p_{k_1+1} \le p_{k_2}$, we have $n_{k_1} \le p_{k_1+1}-1 < p_{k_2} \le n_{k_2}$, so $n_{k_1} \ne n_{k_2}$.
Thus, if $S$ is infinite, there are infinitely many $n$ satisfying the condition.

We now prove that $S$ is infinite.
Assume for contradiction that $S$ is finite. Then there exists an integer $K$ such that for all $k \ge K$, $p_{k+1}-p_k < k$. This implies $p_{k+1}-p_k \le k-1$ for $k \ge K$.
Let $m > K$. We can write $p_m$ as $p_K+\sum_{j=K}^{m-1}(p_{j+1}-p_j)$.
$p_m \le p_K+\sum_{j=K}^{m-1}(j-1)=p_K+\sum_{l=K-1}^{m-2}l=p_K+\frac{(m-2)(m-1)}{2}-\frac{(K-2)(K-1)}{2}$.
$p_m \le \frac{m^2-3m+2}{2}+C_K$, where $C_K=p_K-\frac{(K-2)(K-1)}{2}$ is a constant.
$p_m \le \frac{1}{2}m^2-\frac{3}{2}m+C_K$.
Dividing by $m^2$, we get $\frac{p_m}{m^2} \le \frac{1}{2}-\frac{3}{2m}+\frac{C_K}{m^2}$.
Taking the limit superior as $m \to \infty$, we get $\limsup_{m \to \infty} \frac{p_m}{m^2} \le \frac{1}{2}$.

""",
            "question": "For a positive integer $n$, let $A_{n}$ be the number of perfect powers less than or equal to $n$. Here, a perfect power is a number that can be expressed in the form $a^{b}$, where $a$ is a positive integer and $b$ is an integer greater than or equal to 2. Prove that there are infinitely many $n$ such that $A_{n}$ divides $n+2024$.",
            "is_proof": True,
            "reviewer": "naive",
            "reviews": reviews,
            "label": None,
            "points": None
        }
    return {
        "response": "The answer is 42.",
        "label": ["42"],
        "question": "What is 6 * 7?",
        "use_xverify": False,
    }


def main():
    args = parse_args()
    url = f"{args.host}:{args.port}/"
    payload = build_payload(args.proof, args.reviews)

    print(f"POST {url}")
    print("Payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    resp = requests.post(url, json=payload, timeout=60)
    print(f"\nStatus: {resp.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    except Exception:
        print("Raw response:")
        print(resp.text)


if __name__ == "__main__":
    main()
