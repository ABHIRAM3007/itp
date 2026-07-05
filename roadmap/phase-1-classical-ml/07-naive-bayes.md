# 1.7 — Naive Bayes

## 1. Intuition first

Apply Bayes' rule and pretend features are conditionally independent given the class. Absurd assumption, but it works remarkably well for text.

## 2. Why the topic exists

Fast, simple, works with tiny data, great baseline for text classification, and is the canonical example of a **generative** model.

## 3. What problem it solves

Classification when you can estimate $P(\mathbf{x} \mid y)$ per feature easily (word counts, discrete features).

## 4. Mathematics

By Bayes:

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^d P(x_j \mid y)
$$

Predict $\hat y = \arg\max_y P(y) \prod_j P(x_j \mid y)$. Log for stability:

$$
\hat y = \arg\max_y \Big[\log P(y) + \sum_j \log P(x_j \mid y)\Big]
$$

Variants:
- **Multinomial NB**: features are counts (word frequencies).
- **Bernoulli NB**: binary indicators (word present).
- **Gaussian NB**: continuous features modeled as $\mathcal{N}(\mu_{y,j}, \sigma^2_{y,j})$.

**Laplace smoothing** avoids zero probabilities:

$$
P(w \mid c) = \frac{N_{wc} + \alpha}{N_c + \alpha |V|}
$$

## 5. Every formula explained

- Bayes gives posterior; independence assumption factorizes the likelihood.
- Log-sum keeps arithmetic stable when many features are multiplied.
- Laplace smoothing $\alpha$ adds pseudo-counts so unseen words don't nuke a class.

## 6. Variables

| Symbol | Meaning |
|--------|---------|
| $y$ | class label |
| $x_j$ | $j$-th feature |
| $P(y)$ | class prior |
| $\alpha$ | smoothing constant |
| $|V|$ | vocabulary size |

## 7. Algorithm — Multinomial NB training

1. For each class $c$: compute $P(c) = n_c / n$.
2. For each class $c$ and word $w$: $P(w\mid c) = \tfrac{\text{count}(w, c) + \alpha}{\sum_{w'} (\text{count}(w', c) + \alpha)}$.
3. To predict: compute log-posterior, take argmax.

## 8. Simple example

Emails: "buy viagra now" (spam) and "meeting tomorrow" (ham). New email "buy meeting tomorrow" — Naive Bayes multiplies P(word|class) for each class and picks max.

## 9. Real-world example

Spam filters historically. Sentiment analysis baselines. Language identification.

## 10. Diagram

```mermaid
flowchart LR
    X["Features x"] --> A["Compute P(y) · ΠP(xj|y) for each class"]
    A --> B["Argmax → predicted class"]
```

## 11. Implementation from scratch

```python
import numpy as np
from collections import defaultdict

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):                     # X: n × V count matrix
        classes = np.unique(y)
        self.classes = classes
        self.log_prior = np.log([(y == c).mean() for c in classes])
        Xc = np.array([X[y == c].sum(axis=0) for c in classes]) + self.alpha
        self.log_lik = np.log(Xc / Xc.sum(axis=1, keepdims=True))   # k × V
        return self
    def predict(self, X):
        scores = X @ self.log_lik.T + self.log_prior
        return self.classes[np.argmax(scores, axis=1)]
```

## 12. Implementation using libraries

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
vec = CountVectorizer(ngram_range=(1, 2))
Xtr = vec.fit_transform(train_texts)
m = MultinomialNB(alpha=1.0).fit(Xtr, y_train)
```

## 13. Time complexity

Training: O(n × avg-doc-length). Prediction: O(V) per doc (or O(nnz) sparse).

## 14. Space complexity

O(|classes| × V).

## 15. Advantages

- Extremely fast to train and predict.
- Works with tiny data.
- Handles high-dim sparse features (text) naturally.
- Interpretable per-word log-likelihoods.

## 16. Disadvantages

- Independence assumption is false; predicted probabilities poorly calibrated.
- Struggles with correlated features.
- Beaten by logistic regression / transformers with enough data.

## 17. Interview questions

1. Derive the Naive Bayes classifier.
2. Why "naive"?
3. Difference between Multinomial and Bernoulli NB.
4. What is Laplace smoothing and why?
5. Generative vs discriminative — where does NB sit?
6. Why do we predict in log-space?
7. When does NB beat logistic regression? (Small data.)
8. How would you handle continuous features?
9. Why are NB probabilities usually poorly calibrated?
10. How is NB related to LDA topic models?

## 18. Common mistakes

- Zero probabilities without smoothing.
- Using multinomial NB on TF-IDF (works but a bit odd — treat with care).
- Reporting predicted probabilities as calibrated.

## 19. Optimization techniques

- Use sparse matrices.
- Feature selection (chi²) before NB.
- Complement NB for imbalanced multi-class text.

## 20. Coding exercises

1. Implement Bernoulli NB.
2. Implement Gaussian NB.
3. Reproduce a spam filter on the SMS Spam Collection dataset.
4. Compare NB vs LR on 20-newsgroups for varying training-set sizes.

## 21. Mini project

Language-detection classifier on Wikipedia snippets in 10 languages using character n-gram NB.

## 22. Medium project

Sentiment analysis on IMDB reviews; benchmark NB against LR baseline; produce feature-importance report.

## 23. Advanced project

Build a real-time spam filter service (FastAPI + Multinomial NB with online learning) that updates on user "spam" / "not spam" feedback.

## 24. Where it is used in industry

Text baselines, spam, log classification, quick-and-dirty routing.

## 25. How companies use it

- Support ticket routing.
- Fraud rule pre-filters.
- Spam / abuse first-pass filters.

## 26. When NOT to use it

- When features are strongly correlated and abundant labeled data exists — LR / transformers are better.
- Regression tasks.
