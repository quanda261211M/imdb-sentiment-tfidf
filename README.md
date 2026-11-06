# IMDb Sentiment Classification with TF-IDF and SVM

Final project for the Natural Language Processing course (Master of Data Science), Group No. 03.
Instructor: Dr. Nguyen Kiem Hieu.

**Members:** Nguyen Doan Anh Huy, Nguyen Phuc Hung, Dao Anh Quan

We classify English movie reviews as positive or negative with TF-IDF features and support vector machines, and use controlled comparisons to ask three questions:

1. Do bigrams or trigrams help over unigrams?
2. Does an RBF kernel improve enough over a linear kernel to justify its cost?
3. Do text cleaning and removing common words (`movie`, `film`, stop words) make a useful difference?

## Repository contents

| File | Description |
|---|---|
| `imdb_sentiment_tfidf_svm.ipynb` | Main experiment notebook (cuML on Kaggle, 2x T4 GPU), with saved outputs |
| `[GROUP3] NLP_IMDb_Report.pdf` | Full written report |
| `[GROUP3] NLP_FINAL_Revised.pptx` | Presentation slides (with speaker notes) |

## Setup

- **Data:** Large Movie Review Dataset (Maas et al., 2011), loaded as `stanfordnlp/imdb`.
- **Train:** 20,000 reviews (10,000 per class). **Holdout:** 6,000 new reviews (3,000 per class), seed 42.
- **Holdout design:** normalized text hashes remove exact duplicates, and the 6,000 test IDs from the earlier CPU study are reconstructed and excluded. The reconstruction depends on the assumed source ordering, so it is not a fully verified record.
- **Representations (6):** unigram, unigram+bigram, up to trigram, cleaned unigram+bigram, drop `movie`/`film` unigrams, drop all n-grams containing them. The raw representations share a 12,000-feature budget (`min_df = 2`).
- **Features:** smoothed IDF, sublinear TF, L2 normalization. Vocabulary and IDF are fitted inside each training fold.
- **Models:** linear and RBF `SVC`, with C in {0.5, 2, 8} and gamma in {scale, 0.1, 1} for RBF.
- **Selection:** stratified 3-fold CV on positive-class F1 (72 configurations, 216 fits, 12 final refits). The champion is chosen before the holdout is evaluated.

## Main results

| Item | Result |
|---|---|
| CV-selected model | Cleaned unigram+bigram, RBF SVC, C = 2, gamma = scale |
| Holdout accuracy | 89.300% (5,358 correct, 642 wrong) |
| Positive precision / recall / F1 | 88.681% / 90.100% / 89.385% |
| Bigram gain (RBF, over unigram) | +0.913 pp positive F1 (95% interval excludes zero) |
| Trigram gain (over bigram) | +0.005 pp (interval includes zero) |
| RBF vs linear bigram | +0.316 pp (interval includes zero) |
| Cleaning gain | +0.052 pp (interval includes zero) |
| `movie`/`film` removal | No consistent benefit; all intervals include zero |

Earlier CPU study (scikit-learn, different holdout and tuning design, so not directly comparable with the GPU scores): LinearSVC 89.500% accuracy in 8.9 s versus RBF SVC 89.517% in 1,540.0 s (about 173x longer). The same study finds no significant accuracy difference from removing stop words, including when negation and contrast words are kept.

**Limitations.** Small template tests show the model fails on longer negation, explicit contrast and reported speech (0 of 6 pairs fully correct in each of those groups). Manual verification of the diagnostic labels is unfinished. Bootstrap intervals cover holdout sampling only, not retraining variation, and are not adjusted for multiple comparisons.

## Running the notebook

- The notebook is set to `MODE = "full"` and requires a Kaggle environment with **GPU T4 x2**. It stops with an error if two GPUs are not visible.
- Setting `MODE = "smoke"` runs a small sample (400 train / 200 holdout reviews, one C/gamma) to check the workflow. Do not report smoke results.
- Outputs (metrics, predictions, model bundles, figures) are written to a run directory by the notebook.
- **Demo:** the champion and a `raw_w2_linear` baseline are exported for CPU inference. The demo is available as a notebook widget, a CLI and a Streamlit app. It returns a label and a decision margin, which is not a probability.
- **Local retraining** would need replacing cuML with scikit-learn `SVC`. We have no completed local IMDb result and make no claim about local timings.

## References

1. Maas, A. L. et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL HLT, 142-150. https://aclanthology.org/P11-1015/
2. scikit-learn documentation: `TfidfTransformer`, `SVC`, `LinearSVC`.
