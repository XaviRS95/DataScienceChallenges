### Evaluation:

In the originall challenge, the specified metric was ROC_Score, but for this case-study I have preferred to add some narrative:
Imagine that I am a Data Scientist at a bank building this models and I want to choose the most suitable metric for the scenario of classifying a loan request from a client as invalid or approved. For this, it's necessary to understand our current business aim:

#### 🔍 Mistake Trade-off in Loan Default Prediction

| Scenario                                              | Type of Error    | Metric Focus | Risk                        |
|--------------------------------------------------------|------------------|--------------|-----------------------------|
| Predicts **no default**, but they **default**         | False Negative   | Recall       | 💸 High financial loss      |
| Predicts **default**, but they **don't default**       | False Positive   | Precision    | 🙅 Missed revenue opportunity |


#### 🔹 If your priority is avoiding loan defaults at all costs:

- **Recall is more important** — you want to **catch all defaulters**, even if you wrongly reject some non-defaulters.
- ✅ **Use `F2-score`**: weighs **recall** more than precision.

---

#### 🔹 If your priority is avoiding rejecting reliable clients:

- **Precision is more important** — you want to **be sure** that anyone labeled as "defaulter" truly is.
- ✅ **Use `F0.5-score`**: favors **precision**.


After this has been clarified and, for the sake of this scenario, I will go with the more "conservative" approach and use F2-Score. But wait, what is F2-Score?

The **F2-Score** is a variant of the F1-score that gives **more importance to recall** than to precision.

It is useful when **missing positive cases is more costly** than including some false positives.

### 📌 Formula

\[
F2 = \left(1 + 2^2\right) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(2^2 \cdot \text{Precision}) + \text{Recall}}
\]

Where:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **TP** = True Positives
- **FP** = False Positives
- **FN** = False Negatives

---