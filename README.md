# nlp_workflow

A small, runnable codebase for the Practical NLP series. Every setting uses:

> The product works, but the battery is not good.

The purpose is to inspect the information a text pipeline retains **before**
training a customer-review classifier. There is no trained classifier, labeled
dataset, sentiment prediction, or accuracy claim in this project.

## Run locally

Use Python 3.11 or newer. Tested on Python 3.14.6 with scikit-learn 1.9.0.
No API keys, downloaded language corpora, or paid services are needed.

```bash
cd nlp_workflow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python review_features.py
python -m unittest -v
```

On Windows, activate with `.venv\Scripts\activate` instead.

## What the demo proves

| Setting | `not` count | `good` count | `not good` count |
| --- | ---: | ---: | ---: |
| Unigrams, words retained | 1 | 1 | 0 |
| Unigrams, English stop words removed | 0 | 1 | 0 |
| Unigrams + bigrams, words retained | 1 | 1 | 1 |
| Unigrams + bigrams, English stop words removed | 0 | 1 | 0 |

A unigram is a single-word feature. A bigram is a pair of consecutive tokens.
Using `(1, 2)` includes **both** individual words and two-token phrases.
Zero in the table means that feature is absent, not a sentiment score.

`CountVectorizer` lowercases this text and removes punctuation while tokenizing.
Its English stop-word list removes `not`. Stop-word filtering happens **before**
forming n-grams: adding bigrams afterward cannot recover the removed word.
Filtering also changes adjacency: `battery good` becomes a bigram in the
filtered text even though those words were not consecutive in the original.

The tests additionally derive an affirmative version by removing `not` from the
same review. The two reviews produce identical unigram vectors with English
stop-word removal, but different vectors when words are retained.

## Boundaries

- These checks demonstrate feature behavior, not an improvement in accuracy.
- Bigrams capture local token order, not full sentence meaning or negation scope.
- This is a mixed review: the product works, but the battery is criticized.
  This demo does not assign an overall positive/negative label.
- No stemming or lemmatization is applied. Neither is required for this test.
- Retaining all stop words is a demonstration setting, not a universal rule.
- Real classifier evaluation requires representative labeled data and a held-out
  test set. Fit preprocessing on the training split only to avoid leakage.

## Files

- `review_features.py`: reusable inspection function and command-line demo.
- `test_review_features.py`: nine tests for tokens, counts, negation, and adjacency.
- `requirements.txt`: pinned scikit-learn version.
- `linkedin-post.txt`: short caption based on the verified output.

Reference: [scikit-learn CountVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

Only this project's source files belong in the GitHub repository. Do not upload
the surrounding workspace or the separate portfolio backups.
