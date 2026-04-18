---

# 📊 Supported Dataset Format

> 🚨 **Important:** Using the correct dataset format is critical for the application to work properly.
> Please review the guidelines below before uploading your file.

---

## 📦 Basic Requirements

```
✔ File type: CSV (.csv)
✔ First row must contain column headers
✔ Clean, tabular structure (rows = records, columns = features)
```

---

## 🧩 Data Structure

### ✅ Supported Format

| Age | Salary | Experience | Purchased |
| --- | ------ | ---------- | --------- |
| 25  | 40000  | 2          | No        |
| 32  | 60000  | 5          | Yes       |

---

### ❌ Not Supported

```
✘ Images or image datasets
✘ JSON / nested data
✘ Unstructured text files
✘ Excel files with merged cells
```

---

## 🎯 Target Column (Required)

> 💡 Your dataset **must contain one clear target column**

* 📌 Classification → Categorical values (Yes/No, A/B/C)
* 📌 Regression → Numeric values (Price, Score)

```
✔ One target column only
✘ Multiple targets (not supported)
```

---

## 🔢 Feature Guidelines

```
✔ Numeric columns (preferred)
✔ Limited categorical columns
✔ Simple values (e.g., Gender, City)
```

🚫 Avoid:

* Too many unique values (like IDs)
* Mixed data types in a single column

---

## ⚠️ Missing Values

> The system can handle missing data — but keep it reasonable

```
✔ Small amount of missing values → OK
✘ More than 50% missing → Not recommended
```

---

## 📏 Dataset Size

```
✔ Recommended: 100 – 50,000 rows
⚠ Large datasets may be automatically sampled
```

---

## 🧼 Column Naming Rules

```
✔ age, salary, house_price
✘ Age (Years), $Salary, @price!
```

👉 Keep names:

* Simple
* Clean
* Lowercase (preferred)

---

## 🚫 Common Issues to Avoid

```
✘ Multiple target columns
✘ Unstructured data
✘ Too many categorical values
✘ Empty or corrupted rows
```

---

## 💡 Best Use Cases

✔ House price prediction
✔ Customer churn classification
✔ Student performance analysis
✔ Sales prediction

---

## 🚷 Not Suitable For

```
✘ Image datasets (CNN required)
✘ NLP / text-heavy datasets
✘ Time-series without preprocessing
✘ Multi-table relational data
```

---

## 🧠 Pro Tip

> Clean your dataset before uploading for best results.

* Remove unnecessary columns (IDs, timestamps)
* Handle obvious errors
* Keep only relevant features

---


