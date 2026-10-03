"""Regression tests for the same review used throughout the NLP series."""

import unittest

from sklearn.feature_extraction.text import CountVectorizer

from review_features import REVIEW, inspect_review


class ReviewFeatureTests(unittest.TestCase):
    def test_unigrams_retain_negation_but_not_phrase_features(self):
        result = inspect_review()
        self.assertEqual(result.counts["not"], 1)
        self.assertEqual(result.counts["good"], 1)
        self.assertNotIn("not good", result.counts)
        self.assertEqual(result.counts["the"], 2)

    def test_english_stop_words_remove_not(self):
        result = inspect_review(remove_stop_words=True)
        self.assertNotIn("not", result.tokens)
        self.assertNotIn("not", result.counts)
        self.assertEqual(result.counts["good"], 1)

    def test_bigrams_preserve_not_good_when_words_are_retained(self):
        result = inspect_review(include_bigrams=True)
        self.assertEqual(result.counts["not good"], 1)
        self.assertEqual(result.counts["product works"], 1)
        self.assertEqual(result.counts["the"], 2)

    def test_bigrams_cannot_restore_removed_negation(self):
        result = inspect_review(remove_stop_words=True, include_bigrams=True)
        self.assertNotIn("not good", result.counts)
        self.assertNotIn("not", result.counts)

    def test_removal_changes_which_tokens_are_adjacent(self):
        kept = inspect_review(include_bigrams=True)
        removed = inspect_review(remove_stop_words=True, include_bigrams=True)
        self.assertNotIn("battery good", kept.counts)
        self.assertIn("battery good", removed.counts)

    def test_negated_and_affirmative_versions_collide_after_removal(self):
        affirmative = REVIEW.replace("not good", "good")
        removed = CountVectorizer(stop_words="english")
        rows = removed.fit_transform([REVIEW, affirmative]).toarray()
        self.assertTrue((rows[0] == rows[1]).all())
        kept = CountVectorizer(stop_words=None)
        rows = kept.fit_transform([REVIEW, affirmative]).toarray()
        self.assertFalse((rows[0] == rows[1]).all())

    def test_lowercase_and_punctuation_are_handled(self):
        result = inspect_review()
        self.assertEqual(result.tokens,
                         ("the", "product", "works", "but", "the", "battery", "is", "not", "good"))

    def test_invalid_review_is_rejected(self):
        for review in ("", "   ", None):
            with self.subTest(review=review), self.assertRaises(ValueError):
                inspect_review(review)

    def test_punctuation_only_has_no_vocabulary(self):
        with self.assertRaises(ValueError):
            inspect_review("!!!")


if __name__ == "__main__":
    unittest.main()
