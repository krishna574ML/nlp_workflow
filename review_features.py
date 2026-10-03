"""Inspect the information CountVectorizer keeps from one customer review."""

from dataclasses import dataclass

from sklearn.feature_extraction.text import CountVectorizer


REVIEW = "The product works, but the battery is not good."


@dataclass(frozen=True)
class FeatureResult:
    label: str
    tokens: tuple[str, ...]
    counts: dict[str, int]


def inspect_review(
    review: str = REVIEW,
    *,
    remove_stop_words: bool = False,
    include_bigrams: bool = False,
) -> FeatureResult:
    """Return real tokenization and feature counts, not sentiment predictions.

    English stop-word removal runs before n-gram construction. Consequently,
    pairs in that mode are adjacent in the filtered tokens, not necessarily
    adjacent in the original review.
    """
    if not isinstance(review, str) or not review.strip():
        raise ValueError("review must be a non-empty string")

    vectorizer = CountVectorizer(
        stop_words="english" if remove_stop_words else None,
        ngram_range=(1, 2) if include_bigrams else (1, 1),
    )
    matrix = vectorizer.fit_transform([review])
    names = vectorizer.get_feature_names_out().tolist()
    counts = dict(zip(names, map(int, matrix.toarray()[0]), strict=True))
    # With default word analysis, preprocessing lowercases and tokenization
    # extracts words. No stemming, lemmatization, or model fitting is performed.
    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()
    tokens = tokenize(preprocess(review))
    stop_words = vectorizer.get_stop_words()
    if stop_words is not None:
        tokens = [token for token in tokens if token not in stop_words]

    label = "Unigrams + bigrams" if include_bigrams else "Unigrams"
    label += " / English stop words removed" if remove_stop_words else " / Words retained"
    return FeatureResult(label, tuple(tokens), counts)


def run_demo() -> None:
    print(f"Review: {REVIEW}")
    print("Purpose: inspect text features before training a review classifier.")
    for include_bigrams, remove_stop_words in (
        (False, False), (False, True), (True, False), (True, True)
    ):
        result = inspect_review(
            include_bigrams=include_bigrams,
            remove_stop_words=remove_stop_words,
        )
        print(f"\n{result.label}")
        print("Tokens:", " | ".join(result.tokens))
        print("Feature counts:", result.counts)
        print("Selected features:", {term: result.counts.get(term, 0)
                                    for term in ("not", "good", "not good")})
    print("\nTakeaway: retain negation before forming phrase features.")
    print("Bigrams encode local token order; this demo does not predict sentiment.")


if __name__ == "__main__":
    run_demo()
