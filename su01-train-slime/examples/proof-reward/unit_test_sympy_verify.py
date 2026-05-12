import unittest

from p1 import are_equal_under_sympy, grade_answer_sympy, grade


class TestSympyVerifyRegression(unittest.TestCase):
    def test_integer_mismatch_should_not_match(self):
        """Test that different integers are not incorrectly judged as equal."""
        mismatch_cases = [
            ("180", "184"),
            ("496503", "496524"),
            ("36", "40"),
            ("26", "30"),
            ("7657", "8000"),
            ("30", "26"),
            ("21", "20"),
            ("1431655765", "1431655764"),
        ]
        for gold, pred in mismatch_cases:
            with self.subTest(gold=gold, pred=pred):
                self.assertFalse(
                    are_equal_under_sympy(gold, pred),
                    msg=f"Unexpected sympy equivalence for {gold} vs {pred}",
                )
                is_correct, _, _ = grade_answer_sympy(pred, gold)
                self.assertFalse(
                    is_correct,
                    msg=f"grade_answer_sympy incorrectly accepted {pred} for {gold}",
                )
                is_correct, _, _, _ = grade(pred, gold, False)
                self.assertFalse(
                    is_correct,
                    msg=f"grade incorrectly accepted {pred} for {gold}",
                )

    def test_fraction_vs_integer_mismatch(self):
        """Test that non-equivalent fraction vs integer are rejected."""
        mismatch_cases = [
            ("512578", "4100625/8"),
            ("512578", "2025**2/8"),
            ("3736", "239045/64"),
            ("3736", "1075705/288"),
            ("498", "500"),
        ]
        for gold, pred in mismatch_cases:
            with self.subTest(gold=gold, pred=pred):
                is_correct, _, _ = grade_answer_sympy(pred, gold)
                self.assertFalse(
                    is_correct,
                    msg=f"grade_answer_sympy incorrectly accepted {pred} for {gold}",
                )
                is_correct, _, _, _ = grade(pred, gold, False)
                self.assertFalse(
                    is_correct,
                    msg=f"grade incorrectly accepted {pred} for {gold}",
                )

    def test_symbolic_expression_mismatch(self):
        """Test that non-equivalent symbolic expressions are rejected."""
        mismatch_cases = [
            ("2**(u - 2)", "2**(u - 1) - 1"),
            ("x+1", "x+2"),
            ("n**2", "n**2+1"),
        ]
        for gold, pred in mismatch_cases:
            with self.subTest(gold=gold, pred=pred):
                is_correct, _, _ = grade_answer_sympy(pred, gold)
                self.assertFalse(
                    is_correct,
                    msg=f"grade_answer_sympy incorrectly accepted {pred} for {gold}",
                )
                is_correct, _, _, _ = grade(pred, gold, False)
                self.assertFalse(
                    is_correct,
                    msg=f"grade incorrectly accepted {pred} for {gold}",
                )

    def test_true_equivalent_cases(self):
        """Test that truly equivalent expressions are still accepted."""
        equivalent_cases = [
            ("1/2", "0.5"),
            ("0.60000", "0.6"),
            ("4100625/8", "2025**2/8"),
            ("2*(x+1)", "2*x+2"),
            ("100", "100.0"),
            ("180", "180.00"),
            ("184", "184"),
            ("x**2 - 1", "(x-1)*(x+1)"),
        ]
        for gold, pred in equivalent_cases:
            with self.subTest(gold=gold, pred=pred):
                self.assertTrue(
                    are_equal_under_sympy(gold, pred),
                    msg=f"Expected sympy equivalence for {gold} vs {pred}",
                )
                is_correct, _, _ = grade_answer_sympy(pred, gold)
                self.assertTrue(
                    is_correct,
                    msg=f"grade_answer_sympy should accept {pred} for {gold}",
                )
                is_correct, _, _, _ = grade(pred, gold, False)
                self.assertTrue(
                    is_correct,
                    msg=f"grade should accept {pred} for {gold}",
                )


if __name__ == "__main__":
    unittest.main()
