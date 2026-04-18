# 📊 Dataset Guidelines

> ⚠️ **Important:** Please follow these guidelines to avoid errors while using the application.

---

## 📦 Basic Requirements

* File type must be **CSV (.csv)**
* First row must contain **column headers**
* Data must be in **tabular format** (rows = records, columns = features)

---

## 🧩 Supported Data Format

| Age | Salary | Experience | Purchased |
| --- | ------ | ---------- | --------- |
| 25  | 40000  | 2          | No        |
| 32  | 60000  | 5          | Yes       |

---

## ❌ Not Supported

* Images or image datasets
* JSON or nested data
* Unstructured text files
* Excel files with merged cells

---

## 🎯 Target Column (Required)

Your dataset must contain **one clear target column**.

* **Classification:** Categorical values (Yes/No, A/B/C)
* **Regression:** Numeric values (Price, Score)

**Rules:**

* Only one target column
* Multiple targets are not supported

---

## 🔢 Feature Guidelines

* Numeric columns are preferred
* Categorical columns should be limited
* Use simple values (e.g., Gender, City)

**Avoid:**

* Too many unique values (like IDs)
* Mixed data types in a single column

---

## ⚠️ Missing Values

* Small number of missing values → acceptable
* Too many missing values (>50%) → not recommended

---

## 📏 Dataset Size

* Recommended: **100 to 50,000 rows**
* Large datasets may be sampled automatically

---

## 🧼 Column Naming Rules

**Good:**

* age
* salary
* house_price

**Avoid:**

* Age (Years)
* $Salary
* @price!

Keep names:

* Simple
* Clean
* Lowercase preferred

---

## 🚫 Common Issues

* Multiple target columns
* Unstructured or messy data
* Too many categorical values
* Empty or corrupted rows

---

## 💡 Best Use Cases

* House price prediction
* Customer churn classification
* Student performance analysis
* Sales prediction

---

## 🚷 Not Suitable For

* Image datasets
* NLP / text-heavy datasets
* Time-series without preprocessing
* Multi-table relational data

---

## 🧠 Pro Tip

Clean your dataset before uploading:

* Remove unnecessary columns (IDs, timestamps)
* Fix obvious errors
* Keep only relevant features

---

