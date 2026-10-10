# Human Activity Recognition — Unsupervised Clustering

An unsupervised-learning project that investigates whether smartphone
accelerometer/gyroscope features can reveal meaningful human-activity
groups **without using activity labels to train the clustering
models**. Labels are used only afterwards, to interpret and score the
clusters that were found.

## Dataset

UCI **Human Activity Recognition Using Smartphones**: 10,299 recordings
(7,352 train + 2,947 test) from 30 volunteers wearing a waist-mounted
Samsung Galaxy S II, performing six activities — WALKING,
WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING. Each
recording is a 2.56-second window (128 readings, 50% overlap, 50 Hz)
summarised into 561 time/frequency-domain features. No missing values.

Files expected in the project root: `train.csv`, `test.csv`.

## Methodology

1. **Inspection & quality checks** — shape, dtypes, missing values,
   duplicates, near-constant features, feature scale/variance spread.
2. **Preprocessing** — separate the 561-feature matrix from
   `subject`/`Activity`/`Split`; `Activity` is held out of every
   clustering/PCA fit.
3. **EDA** — feature distributions, single-feature separability,
   correlation structure, one multivariate scatter — each with an
   Observation + Inference.
4. **Scaling** — `StandardScaler` (feature variances differ ~340x even
   though ranges look similar).
5. **PCA** — 65 components retain 90% of variance (PC1 alone explains
   ~51%); a 2-component projection is kept only for visualisation.
6. **Clustering** — K-Means and Agglomerative (Ward) Clustering, each
   evaluated across k = 2..10 using the Elbow method, Silhouette Score,
   Calinski-Harabasz and Davies-Bouldin indices; DBSCAN is explored as
   an optional density-based extension with an eps/min_samples grid
   search guided by a k-distance graph.
7. **Evaluation** — internal metrics first (label-free), then external
   agreement (Adjusted Rand Index, Normalized Mutual Information,
   cluster-vs-activity crosstabs) computed only after clustering.
8. **Interpretation & final model** — Agglomerative Clustering at
   k = 6 is selected as the final solution (see notebook Section 24 for
   the full justification).

## Key Findings

- The strongest **purely data-driven** structure is a **2-cluster**
  split (moving vs. stationary activities) — every internal metric
  (Silhouette, Calinski-Harabasz, Davies-Bouldin) favours k = 2 over
  k = 6.
- The **best internal score does not produce the best external
  agreement**: Agglomerative Clustering at k = 6 has the highest
  ARI/NMI against the real activities of every model tested, despite
  scoring lower internally than the k = 2 solutions.
- **LAYING** is the most cleanly separable activity (one cluster
  reaches ~96% purity). **SITTING vs. STANDING** is the hardest pair to
  separate — both are static, upright postures whose main sensor
  difference is a subtle gravity angle rather than motion.
- DBSCAN is highly sensitive to its `eps`/`min_samples` parameters in
  this feature space, suggesting the data lacks the sharp density gaps
  DBSCAN needs to shine here.

See `notebook/human_activity_clustering.ipynb` for the full analysis,
plots, and cluster-by-cluster interpretation.

## Repository Structure

```
human-activity-clustering/
├── notebook/
│   └── human_activity_clustering.ipynb
├── outputs/
│   └── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook notebook/human_activity_clustering.ipynb
```

## Limitations

See the notebook's final section ("Conclusions and Limitations") for a
full discussion — in short: cluster boundaries reflect what the
feature engineering emphasises (motion, not fine posture), a 2D PCA
plot keeps only ~57% of total variance, Ward-linkage clustering assumes
roughly convex/similarly-sized clusters, and per-subject differences in
how activities were performed were not investigated separately.
