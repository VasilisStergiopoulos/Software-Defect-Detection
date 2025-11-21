# 📘 Software Defect Detection  
### Predicting defective software modules using machine learning & cross-validation

This project was developed as part of an academic assignment on **Software Defect Detection**, a classic problem in software engineering. The goal is to classify software modules as **defective** or **non-defective** based on static code metrics.

We evaluate **7 machine learning algorithms** across **3 datasets** using **5-fold Stratified Cross-Validation**, comparing multiple normalization methods and performance metrics tailored for imbalanced data.

---

## 📂 Datasets

The project uses three well-known software defect datasets:

| Dataset | Source | Target Column |
|---------|--------|----------------|
| **JM1** | NASA / PROMISE Repository | `defects` |
| **MC1** | NASA / PROMISE Repository | `c` |
| **PC3** | NASA / PROMISE Repository | `c` |

Each dataset contains:
- Software metrics (e.g., LOC, Halstead metrics, complexity, branch count)
- A binary defect label indicating whether a module contains bugs

> ⚠️ Datasets are **not included** in this repo due to licensing restrictions.  
>  You can download them from the PROMISE repository or Kaggle.

---

## 🧠 Objectives

For each dataset, we aim to:

1. Train multiple machine learning algorithms  
2. Apply three normalization strategies  
3. Use **5-fold Stratified Cross-Validation** for reliable evaluation  
4. Measure performance with:
   - **Accuracy**
   - **Weighted F1-score**
   - **Geometric Mean (G-Mean)** – crucial for imbalanced datasets
   - **Fit time**

5. Compare results through bar-plot visualizations  
6. Provide written analysis of model performance per dataset

---

## 🔧 Machine Learning Models

The following **7 classifiers** were evaluated:

- **Logistic Regression**
- **Perceptron**
- **Decision Tree**
- **Random Forest**
- **Linear SVM (LinearSVC)**
- **RBF SVM (SVC with RBF kernel)**
- **Feed-Forward Neural Network (MLPClassifier)**

---

## ⚙️ Normalization Methods

Each classifier was tested with three preprocessing configurations:

1. **No normalization**
2. **Min-Max Scaling**
3. **Standard Scaling**

Tree-based models (Decision Tree, Random Forest) are invariant to scaling,  
while linear/SVM/NN models benefit significantly from normalization.

---

## 🔍 Evaluation Metrics

We evaluate each (model × normalization) combination using:

### ✔ Accuracy  
Overall classification correctness.

### ✔ Weighted F1-Score  
Balances precision & recall across classes, weighted by class frequency.

### ✔ G-Mean (Geometric Mean Score)  
A strong indicator for **imbalanced datasets**, ensuring both classes are predicted well.

### ✔ Fit Time  
Total training + validation time across the 5 CV folds.

---

## 🔄 Cross-Validation Strategy

We use:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Stratified cross-validation is used because:

- Many defect datasets are **imbalanced**
- It ensures each fold preserves **original class ratios**
- It produces **more reliable and fair comparisons** across classifiers

---

## 📊 Visualizations

For each dataset (`jm1`, `mc1`, `pc3`), the notebook produces visualizations:

- A figure containing **4 bar charts** comparing:
  - **Accuracy**
  - **F1-score**
  - **G-Mean**
  - **Fit time**

Each chart shows results for:

- **No normalization**
- **Min-Max scaling**
- **Standard scaling**

across all **7 classifiers**.


---

## 🧪 Experimental Workflow

The notebook follows this pipeline:

1. **Load datasets**
2. **Clean / convert numeric columns**  
   (JM1 required specific type conversion due to inconsistent numeric types)
3. **Split into features (X) and labels (y)**
4. **Loop through each dataset**:
   - Loop through each classifier
   - Loop through each normalization method
     - Build the pipeline (**normalizer → classifier**)
     - Perform **5-fold stratified cross-validation**
     - Record **mean metrics** across folds
     - Track **fit time**
5. **Generate bar-plot comparisons** for accuracy, F1-score, G-Mean, and fit time
6. **Provide written analysis** for each dataset
