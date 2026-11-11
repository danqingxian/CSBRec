"""Regression checks for evaluation candidate sampling."""
import unittest

import numpy as np

from csbrec.data import sample_candidates


class CandidateSamplingTests(unittest.TestCase):
    def test_seeded_candidates_match_existing_protocol(self):
        seen = [[1, 2, 3], [4, 5, 6]]
        targets = [3, 6]
        result = sample_candidates(seen, targets, 20, 5, np.random.default_rng(42))
        np.testing.assert_array_equal(result, [[3, 16, 14, 9, 18, 5], [6, 11, 20, 15, 16, 3]])

    def test_target_is_excluded_even_if_missing_from_history(self):
        result = sample_candidates([[1]], [2], 5, 3, np.random.default_rng(42))
        self.assertEqual(result[0, 0], 2)
        self.assertEqual(set(result[0, 1:]), {3, 4, 5})

    def test_impossible_request_fails_before_consuming_randomness(self):
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(ValueError, "only 1 unseen items"):
            sample_candidates([[1, 2, 3]], [3], 4, 2, rng)
        self.assertEqual(rng.integers(10000), np.random.default_rng(42).integers(10000))

    def test_duplicates_and_padding_do_not_reduce_available_catalogue(self):
        result = sample_candidates([[0, 1, 1, 99]], [2], 4, 2, np.random.default_rng(42))
        self.assertEqual(set(result[0, 1:]), {3, 4})

    def test_zero_negatives_and_empty_users(self):
        np.testing.assert_array_equal(sample_candidates([[1]], [1], 1, 0, np.random.default_rng(42)), [[1]])
        self.assertEqual(sample_candidates([], [], 1, 0, np.random.default_rng(42)).shape, (0, 1))

    def test_invalid_parameters(self):
        for item_num, n_neg in [(0, 1), (2.5, 1), (True, 0), (4, -1), (4, 1.5), (4, True)]:
            with self.subTest(item_num=item_num, n_neg=n_neg), self.assertRaises(ValueError):
                sample_candidates([[1]], [1], item_num, n_neg, np.random.default_rng(42))
        with self.assertRaisesRegex(ValueError, "same length"):
            sample_candidates([], [1], 4, 1, np.random.default_rng(42))
        for target in [0, 5, 1.5, True]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                sample_candidates([[]], [target], 4, 1, np.random.default_rng(42))


if __name__ == "__main__":
    unittest.main()
