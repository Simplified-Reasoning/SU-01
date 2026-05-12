"""
Quick manual test for the proof verifier.

Prereqs:
- A local OpenAI-compatible model server running on MODEL_PORT (default 34882).
- The proof_verifier module in the same directory.

Run:
    python test_proof.py                    # 运行所有测试
    python test_proof.py --mode standard    # 只测试 standard 模式
    python test_proof.py --mode marking     # 只测试 marking 模式
"""
import os
import argparse

from proof_verifier import compute_score_proof


def test_standard_mode(model_port: int):
    """测试标准模式"""
    print("\n" + "=" * 60)
    print("=== Standard Mode Test ===")
    print("=" * 60)

    problem = "Prove that the sum of the interior angles of any triangle is 180 degrees."
    proof = """
    We can place the triangle ABC in the plane. Extend side BC to a line.
    The exterior angle at C equals the sum of the two remote interior angles (A and B).
    Also, interior angle at C plus exterior angle at C is a straight angle (180 degrees).
    Therefore, angle A + angle B + angle C = 180 degrees.
    """

    result = compute_score_proof(
        proof_output=proof,
        problem=problem,
        reviewer="pessimistic",  # or "standard"
        reviews=2,
        model_port=model_port,
    )

    print(f"Model port: {model_port}")
    print(f"Score: {result['score']}")
    print(f"Acc: {result['acc']}")
    print(f"Scored by: {result['scored_by']}")
    print("\nReview text:")
    print(result["review"][:500] + "..." if len(result["review"]) > 500 else result["review"])


def test_marking_mode(model_port: int):
    """测试 marking 模式（IMO 风格，总分 7 分）"""
    print("\n" + "=" * 60)
    print("=== Marking Mode Test (IMO-style, 7 points total) ===")
    print("=" * 60)

    # 使用 IMO 风格的问题和 marking 标准
    problem = "Can the incenters of the four faces of a tetrahedron be coplanar?"
    
    # 模拟学生的解答（包含部分正确的步骤）
    proof = """
    Let the tetrahedron have vertices A, B, C, D. We assign coordinates:
    - A = (0, 0, 0)
    - B = (a, 0, 0) for some a > 0
    - C = (0, b, 0) for some b > 0
    - D = (0, 0, c) for some c > 0
    
    The incenter of a triangle with vertices P, Q, R and opposite side lengths p, q, r is given by:
    I = (p·P + q·Q + r·R) / (p + q + r)
    
    For face BCD (opposite to A), let the side lengths be:
    - |CD| = sqrt(b² + c²)
    - |BD| = sqrt(a² + c²)  
    - |BC| = sqrt(a² + b²)
    
    The incenter I_A of face BCD is:
    I_A = (|CD|·B + |BD|·C + |BC|·D) / (|CD| + |BD| + |BC|)
    
    Similarly, we can compute the incenters I_B, I_C, I_D for the other three faces.
    
    For four points to be coplanar, the determinant of the matrix formed by three vectors 
    connecting one point to the other three must be zero.
    
    After computation, this determinant is generally non-zero for a non-degenerate tetrahedron.
    Therefore, the four incenters cannot be coplanar.
    """
    
    # IMO 风格的 marking 标准（总分 7 分）
    marking = [
        "Award 0.5 pt for correctly stating the incenter of a triangular face as a convex combination of its vertices weighted by opposite side lengths; otherwise 0 pt.",
        "Award 1.0 pt for assigning coordinates to the tetrahedron vertices (e.g., (0,0,0), (a,0,0), (0,b,0), (0,0,c) with a,b,c>0); otherwise 0 pt.",
        "Award 2.0 pt for deriving explicit coordinate expressions for all four incenters in terms of edge lengths; otherwise 0 pt.",
        "Award 2.0 pt for constructing and expanding the determinant condition for coplanarity of the four incenters; otherwise 0 pt.",
        "Award 1.5 pt for proving the determinant is strictly non-zero under non-degenerate tetrahedron constraints; otherwise 0 pt."
    ]

    result = compute_score_proof(
        proof_output=proof,
        problem=problem,
        marking=marking,  # 传入 marking 参数会自动使用 marking 模式
        model_port=model_port,
    )

    print(f"Model port: {model_port}")
    print(f"\n--- Scores ---")
    print(f"Total Score: {result.get('total_score', 'N/A')} / {result.get('max_score', 'N/A')}")
    print(f"Normalized Score: {result['score']:.4f}")
    print(f"Acc: {result['acc']}")
    print(f"Scored by: {result['scored_by']}")
    print(f"\n--- Marking Criteria ---")
    for i, criterion in enumerate(marking, 1):
        print(f"  {i}. {criterion}")
    print(f"\n--- Review (truncated) ---")
    review = result["review"]
    print(review)


def main():
    parser = argparse.ArgumentParser(description="Test proof verifier")
    parser.add_argument("--mode", choices=["standard", "marking", "all"], default="all",
                        help="Test mode: standard, marking, or all (default: all)")
    parser.add_argument("--port", type=int, default=35999,
                        help="Model server port (default: 35999)")
    args = parser.parse_args()

    model_port = args.port

    if args.mode == "standard":
        test_standard_mode(model_port)
    elif args.mode == "marking":
        test_marking_mode(model_port)
    else:  # all
        test_standard_mode(model_port)
        test_marking_mode(model_port)

    print("\n" + "=" * 60)
    print("=== Tests Complete ===")
    print("=" * 60)


if __name__ == "__main__":
    main()
