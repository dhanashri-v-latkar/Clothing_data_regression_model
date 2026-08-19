# H&M Customer Review Sentiment Analysis

## 📌 Project Overview

This project performs **sentiment analysis on H&M customer reviews** using Natural Language Processing (NLP) and Machine Learning.

The goal is to process customer comments, clean and transform the textual data into numerical features using **TF-IDF**, and classify the sentiment of each review using **Logistic Regression**.

The dataset contains customer reviews along with their associated themes, sentiment labels, and timestamps.

---

## 🎯 Objectives

* Clean and preprocess customer review text.
* Identify the language of each review.
* Keep only English-language reviews.
* Apply NLP preprocessing techniques such as:

  * Lowercasing
  * Removing digits
  * Lemmatization
  * Stop-word removal
* Convert text into numerical features using **TF-IDF**.
* Train a **Logistic Regression** sentiment classification model.
* Evaluate the model's predictions.

---

## 📊 Dataset

The project uses the `hm.csv` dataset.

The dataset contains the following columns:

| Column      | Description                  |
| ----------- | ---------------------------- |
| `comment`   | Customer review text         |
| `theme`     | Theme/category of the review |
| `sentiment` | Sentiment label              |
| `datetime`  | Date and time of the review  |

The sentiment values in the dataset are:

* `-1` → Negative
* `0` → Neutral
* `1` → Positive

---

## 🔄 Project Workflow

```text
H&M Customer Reviews
        ↓
Load Dataset
        ↓
Remove Missing Values
        ↓
Remove Very Short Comments
        ↓
Detect Language
        ↓
Keep English Reviews
        ↓
Text Cleaning
        ↓
Lemmatization + Stop-word Removal
        ↓
Train/Test Split
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
Sentiment Prediction
        ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

### 1. Missing Value Removal

Missing records were removed and the dataframe index was reset.

The dataset was reduced to **9,252 records** after removing missing values.

### 2. Remove Short Reviews

Reviews containing 20 characters or fewer were removed.

After this filtering, **8,126 records** remained.

### 3. Language Detection

The `langdetect` library was used to detect the language of each comment.

Only English reviews were retained, resulting in **7,888 reviews**.

### 4. Text Cleaning

spaCy was used for NLP preprocessing.

The cleaning function performs:

* Lowercasing
* Digit removal
* Lemmatization
* Stop-word removal
* Removal of tokens shorter than three characters

Example transformation:

```text
"very good products, great customer service"
                    ↓
"good product great customer service satisfied"
```

---

## 🧠 Feature Engineering — TF-IDF

The cleaned reviews were converted into numerical representations using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

A maximum of **1,000 features** was selected:

```python
TfidfVectorizer(max_features=1000)
```

The training data therefore produced a TF-IDF matrix with shape:

```text
(6310, 1000)
```

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was selected as the classification algorithm.

```python
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train_tfidf, y_train)
```

The model was trained on the TF-IDF representation of the customer reviews.

---

## 📚 Train-Test Split

The dataset was divided into training and testing sets using an 80/20 split.

```text
Training samples: 6310
Testing samples: 1578
```

---

## 📈 Sentiment Distribution

The notebook also examines the distribution of the sentiment classes using a normalized bar chart.

The sentiment classes are represented as:

```text
-1 → Negative
 0 → Neutral
 1 → Positive
```

---

## 📊 Model Prediction

The model generates class probabilities using:

```python
y_pred = log.predict_proba(X_test_tfidf)
```

The notebook then maps the predicted class probabilities back to the sentiment labels `-1`, `0`, and `1`.

---

## ⚠️ Evaluation Note

The notebook currently contains an error during accuracy and confusion-matrix calculation.

`predict_proba()` returns **probabilities**, not final class labels. Passing `y_pred` directly to `accuracy_score()` causes:

```text
ValueError:
Classification metrics can't handle a mix of
continuous-multioutput and multiclass targets
```

The correct approach is to first convert the probabilities into predicted class labels, for example:

```python
y_pred = log.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
```

Then the confusion matrix can be calculated using:

```python
cm = confusion_matrix(y_test, y_pred)
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **spaCy**
* **Langdetect**
* **Scikit-learn**
* **TF-IDF**
* **Logistic Regression**

---

## 📁 Project Structure

```text
H&M-Sentiment-Analysis/
│
├── hm.csv
├── Hm.ipynb
└── README.md
```

---

## 🚀 Future Improvements

* Correct and complete model evaluation.
* Add precision, recall, and F1-score.
* Generate a proper confusion matrix.
* Compare Logistic Regression with models such as:

  * Naive Bayes
  * Linear SVM
  * Random Forest
* Tune TF-IDF parameters.
* Handle sentiment-specific negation more carefully.
* Experiment with advanced NLP models such as BERT.

---

## 📌 Conclusion

This project demonstrates an end-to-end **NLP sentiment classification pipeline** for H&M customer reviews.

The workflow covers data cleaning, language detection, text preprocessing, lemmatization, TF-IDF feature extraction, train-test splitting, and Logistic Regression classification.

The notebook provides a strong foundation for a traditional machine-learning-based sentiment analysis system, with model evaluation requiring a small correction before the final performance can be reliably reported.
