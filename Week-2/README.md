# Assignment 2.1: Text Vectorization Implementation

## Objective
To manually implements the TF-IDF algorithm and compares the results with outputs from scikit-learn’s `CountVectorizer` and `TfidfVectorizer`.

## Corpus
```python
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]
```

## 1. Comparison of Word Scores

### Manual TF-IDF
- Computed using the classic IDF formula:
  \[
  \text{idf}(t) = \log_{10} \left(\frac{N}{df(t)}\right)
  \]
  where:
  - \(N\) is the total number of documents in the corpus
  - \(df(t)\) is the number of documents in which term \(t\) appears

### Scikit-learn’s `CountVectorizer`
- This does **not** compute TF-IDF. It gives the raw count of each word per document.
- Common words like **“the”** will have higher values simply due to frequent occurrence.

### Scikit-learn’s `TfidfVectorizer`
- Uses a **smoothed version** of IDF by default:
  \[
  \text{idf}(t) = \log_{10} \left(\frac{N + 1}{df(t) + 1}\right) + 1
  \]
- This prevents division by zero and ensures even rare terms have non-zero weights.
- To switch this off and use the classic formula, use:
  ```python
  TfidfVectorizer(smooth_idf=False)
  ```

### Output Summary
| Term       | Manual TF-IDF | CountVectorizer | TfidfVectorizer (scikit-learn) |
|------------|----------------|------------------|------------------------------|
| the        | Low score      | High count       | Lower TF-IDF due to smoothing |
| sun/moon   | Moderate score | Appears in 2 docs| Reflected similarly with smoothing |
| celestial  | High score     | Appears once     | High TF-IDF as it's rare |

## 2. Why Scores Differ for Common Words like “the”

- **Manual TF-IDF** penalizes frequent words heavily using the classic formula. So terms like “the” that appear in every document get an IDF of zero.
- **TfidfVectorizer** (with smoothing) ensures no IDF is ever exactly zero. This results in **non-zero weights** for frequent words like “the”.
- **CountVectorizer** doesn’t penalize common words at all—so “the” may dominate unless stopword removal is applied.

## Conclusion
- Smoothing in `TfidfVectorizer` can be useful for numerical stability but may lead to unexpected results for frequent terms.
- Manual TF-IDF gives complete transparency and follows the theoretical formula directly.
