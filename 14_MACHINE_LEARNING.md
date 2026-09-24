# BÀI 1: GIỚI THIỆU TỔNG QUAN VỀ HỌC MÁY (MACHINE LEARNING)

## Mục tiêu học tập
- Hiểu khái niệm Machine Learning và vai trò trong Data Science
- Phân biệt các loại Machine Learning  
- Nắm vững quy trình 10 bước xây dựng mô hình ML
- Hiểu overfitting, underfitting và cách khắc phục
- Có cái nhìn tổng quan trước khi học chi tiết
- Làm quen với thư viện **scikit-learn** – thư viện học máy tiêu chuẩn của Python

> **Cài đặt:** `pip install scikit-learn` (đã có sẵn trong Anaconda và Google Colab).
> Các đoạn code minh họa trong bài dùng bộ dữ liệu **Breast Cancer Wisconsin** có sẵn trong scikit-learn (bài toán phân loại nhị phân: u lành tính/ác tính), không cần tải từ Internet.

---

## 1.1 Machine Learning là gì?

### 1.1.1 Định nghĩa

**Machine Learning (Học máy)** là khả năng máy tính **học từ dữ liệu** để đưa ra dự đoán hoặc quyết định mà không cần lập trình rõ ràng.

**Định nghĩa của Arthur Samuel (1959):**
> "Machine Learning là lĩnh vực nghiên cứu giúp máy tính có khả năng học mà không cần được lập trình một cách tường minh."

**Định nghĩa của Tom Mitchell (1997):**
> "Một chương trình máy tính được cho là học từ kinh nghiệm **E** đối với một số lớp nhiệm vụ **T** và thước đo hiệu suất **P**, nếu hiệu suất của nó trong nhiệm vụ **T**, được đo bằng **P**, cải thiện với kinh nghiệm **E**."

### 1.1.2 So sánh với lập trình truyền thống

**Bảng so sánh:**

| Khía cạnh | Lập trình truyền thống | Machine Learning |
|-----------|------------------------|------------------|
| **Input** | Dữ liệu + **Quy tắc (code)** | Dữ liệu + **Kết quả** |
| **Output** | Kết quả | **Quy tắc (Mô hình)** |
| **Cách tiếp cận** | Lập trình viên viết logic | Máy tự học từ dữ liệu |
| **Thay đổi** | Sửa code | Cung cấp thêm dữ liệu |
| **Phù hợp** | Quy tắc rõ ràng | Quy tắc phức tạp, nhiều biến số |

**Ví dụ: Phát hiện chó trong ảnh**

- **Lập trình truyền thống**: Lập trình viên viết rule: "IF có 4 chân AND có đuôi AND có lông..."  
  ❌ Khó: Không bao quát được mọi trường hợp

- **Machine Learning**: Cho máy xem 10,000 ảnh chó + 10,000 ảnh khác  
  ✅ Máy tự học patterns, xử lý tốt mọi góc độ, ánh sáng

---

## 1.2 Các loại Machine Learning

### Bảng tổng quan

| Loại | Dữ liệu | Mục tiêu | Ví dụ |
|------|---------|----------|-------|
| **Supervised** | Có nhãn (X, y) | Dự đoán y từ X | Dự đoán giá nhà, Phát hiện spam |
| **Unsupervised** | Không nhãn (X) | Tìm cấu trúc | Phân nhóm khách hàng |
| **Reinforcement** | Hành động → Reward | Tối ưu hành động | Game AI, Robot |

### 1.2.1 Supervised Learning (Học có giám sát)

**Regression (Hồi quy) - Dự đoán số liên tục**

| Bài toán | Features (X) | Target (y) |
|----------|--------------|------------|
| Giá nhà | Diện tích, phòng, vị trí | 5 tỷ VNĐ |
| Doanh thu | Chi phí quảng cáo, mùa | 100 triệu |
| Điểm thi | Giờ học, điểm cũ | 8.5 điểm |

**Ví dụ minh họa:**

![Regression Example](images/02_regression_example.png)

*Hình 1: Dự đoán giá nhà dựa trên diện tích - Ví dụ về Regression*

**Classification (Phân loại) - Dự đoán nhãn rời rạc**

| Bài toán | Features (X) | Target (y) | Loại |
|----------|--------------|------------|------|
| Email spam | Nội dung, người gửi | Spam/Ham | Binary |
| Chẩn đoán bệnh | Triệu chứng, xét nghiệm | A/B/C/Khỏe | Multi-class |
| Nhận dạng số | Pixels ảnh | 0-9 | 10 classes |

**Ví dụ minh họa:**

![Classification Example](images/03_classification_example.png)

*Hình 2: Phân loại hoa Iris - Ví dụ về Classification với decision boundary*

### 1.2.2 Unsupervised Learning (Học không giám sát)

**Clustering (Phân cụm)**

Ví dụ: Phân nhóm khách hàng theo tuổi & thu nhập

```
Không có nhãn trước → Máy tự tìm 3 nhóm:
- Nhóm 1: Trẻ, thu nhập thấp  
- Nhóm 2: Trung niên, thu nhập cao
- Nhóm 3: Cao tuổi, thu nhập trung bình
```

**Ví dụ minh họa:**

![Clustering Example](images/04_clustering_example.png)

*Hình 3: Phân cụm khách hàng thành 3 nhóm tự động*

**Dimensionality Reduction (Giảm chiều)**

```
100 features → 10 features chính
→ Giữ được 95% thông tin
→ Training nhanh hơn, dễ visualize
```

---

## 1.3 Quy trình 10 bước xây dựng mô hình ML

### Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────┐
│                  QUY TRÌNH 10 BƯỚC                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Problem Understanding    ┐                          │
│  2. Data Understanding        ├─ Business Understanding │
│  3. Feature Understanding    ┘                          │
│                                                         │
│  4. Feature Engineering      ┐                          │
│  5. Dataset Partition         ├─ Data Preparation       │
│  6. Data Modelling           ┘                          │
│                                                         │
│  7. Data Evaluation          ┐                          │
│  8. Hyper-parameter Tuning    ├─ Model Optimization     │
│  9. Build Pipeline           ┘                          │
│                                                         │
│  10. Conclusion                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Step 1: Problem Understanding

**Xác định bài toán:**
- Loại: Regression? Classification? Clustering?
- Features: Những gì ta có
- Target: Muốn dự đoán gì
- Metric: Accuracy? RMSE? F1?

**Ví dụ:**

| Câu hỏi | Trả lời |
|---------|---------|
| Loại bài toán? | Binary Classification |
| Features? | Thời gian dùng, khiếu nại, giá |
| Target? | Churn (Yes/No) |
| Metric? | F1-score |

### Step 2: Data Understanding

**Kiểm tra 5 vấn đề:**

| Vấn đề | Ví dụ | Nguy hiểm |
|--------|-------|-----------|
| **Missing** | 20% thiếu "Thu nhập" | > 30% |
| **Outliers** | Giá = 1 tỷ khi TB 5 tỷ | Ảnh hưởng lớn |
| **Inconsistent** | Cùng địa chỉ, giá chênh 50% | Cần clean |
| **Imbalanced** | 99% Normal, 1% Fraud | Cần resample |
| **Skewness** | Thu nhập lệch phải | Cần transform |

### Step 3: Feature Understanding (EDA)

Phân tích:
1. **Univariate** (1 biến): Histogram, Stats
2. **Bivariate** (2 biến): Scatter, Correlation  
3. **Multivariate** (Nhiều biến): Heatmap

### Step 4: Feature Engineering

**6 kỹ thuật chính:**

**1. Missing/Outlier Handling**
- Mean/Median Imputation
- Drop nếu > 30%
- IQR method

**2. Feature Transformation**
- Log: Thu nhập lệch → log(thu nhập)
- Square root: Giảm outliers

**3. Feature Enrichment**
```
Gốc: Diện tích=80m², Giá=4tỷ
Mới: Giá/m² = 50 triệu/m²
```

**4. Feature Selection**
- Filter: Correlation
- Wrapper: Forward/Backward
- Embedded: Lasso, Tree importance

**5. Feature Encoding**

| Loại | Trước | Sau | Dùng khi |
|------|-------|-----|----------|
| Label | Red, Green, Blue | 0, 1, 2 | Có thứ tự |
| One-Hot | Red, Green, Blue | [1,0,0], [0,1,0], [0,0,1] | Không thứ tự |

**6. Feature Scaling**

![Feature Scaling](images/07_feature_scaling.png)

*Hình 4: So sánh các phương pháp scaling - Original, Normalization, Standardization*

| Phương pháp | Công thức | Kết quả | Dùng cho |
|-------------|-----------|---------|----------|
| Normalization | (x-min)/(max-min) | [0, 1] | Neural Networks, KNN |
| Standardization | (x-mean)/std | Mean=0, SD=1 | Linear, SVM, PCA |

**Minh họa bằng Python (scikit-learn):**

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Nạp dữ liệu: X là DataFrame các features, y là nhãn (0 = ác tính, 1 = lành tính)
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print(X.shape, y.value_counts().to_dict())

# 1. Missing Handling: điền giá trị thiếu bằng median
imputer = SimpleImputer(strategy="median")

# 2. Feature Transformation: log cho biến lệch phải
area_log = np.log1p(X["mean area"])          # log(1 + x), an toàn khi x = 0

# 3. Feature Enrichment: tạo feature mới từ feature cũ
X_new = X.assign(ratio_area_perimeter=X["mean area"] / X["mean perimeter"])

# 5. Feature Encoding: One-Hot cho biến phân loại
colors = pd.DataFrame({"color": ["Red", "Green", "Blue", "Green"]})
print(pd.get_dummies(colors, columns=["color"], dtype=int))

# 6. Feature Scaling
X_norm = MinMaxScaler().fit_transform(X)     # Normalization → [0, 1]
X_std = StandardScaler().fit_transform(X)    # Standardization → mean 0, SD 1
print("Min/Max sau Normalization:", X_norm.min().round(2), X_norm.max().round(2))
print("Mean/SD sau Standardization:", X_std.mean().round(2), X_std.std().round(2))
```

> ⚠️ Trong dự án thật, các bước `fit` (tính min/max, mean/SD, median...) chỉ được thực hiện trên **tập train**, sau đó mới `transform` cho tập validation/test – nếu không sẽ bị **rò rỉ dữ liệu (data leakage)**. Cách an toàn nhất là dùng **Pipeline** (Step 9).

### Step 5: Dataset Partition

**Train/Validation/Test Split**

![Train Test Split](images/05_train_test_split.png)

*Hình 5: Chia dữ liệu thành Train (70%), Validation (15%), Test (15%)*

```
TOÀN BỘ (100%)
├─ TRAIN (70%): Để học
├─ VALIDATION (15%): Tune hyperparameters  
└─ TEST (15%): Đánh giá cuối (chỉ dùng 1 lần!)
```

**Imbalanced Data:**

```
Vấn đề: 99% Normal, 1% Fraud
→ Model dự đoán tất cả "Normal" → 99% accuracy!
   Nhưng KHÔNG bắt được fraud!

Giải pháp:
- Oversampling: SMOTE
- Undersampling  
- Đổi metric: F1 thay vì Accuracy
```

**Minh họa bằng Python:**

```python
from sklearn.model_selection import train_test_split

# Bước 1: Tách 15% làm TEST
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)   # stratify: giữ tỷ lệ các lớp

# Bước 2: Từ phần còn lại, tách VALIDATION (15% tổng ≈ 0.15/0.85 phần còn lại)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42)

print("Train:", len(X_train), "| Validation:", len(X_val), "| Test:", len(X_test))
print("Tỷ lệ lớp 1 - train/test:", round(y_train.mean(), 3), round(y_test.mean(), 3))
```

> Với dữ liệu mất cân bằng, thư viện `imbalanced-learn` (`pip install imbalanced-learn`) cung cấp `SMOTE` để oversampling.

### Step 6: Data Modelling

**So sánh thuật toán:**

| Bài toán | Đơn giản | Trung bình | Nâng cao |
|----------|----------|------------|----------|
| **Regression** | Linear Regression | Decision Tree | Random Forest, XGBoost |
| **Classification** | Logistic Regression | KNN, SVM | Random Forest, Neural Networks |
| **Clustering** | K-Means | Hierarchical | DBSCAN |

**Chiến lược:** Bắt đầu đơn giản → Tăng dần độ phức tạp

![Model Comparison](images/10_model_comparison.png)

*Hình 6: So sánh accuracy của các thuật toán ML*

### Step 7: Data Evaluation

**Classification Metrics:**

![Confusion Matrix](images/06_confusion_matrix.png)

*Hình 7: Confusion Matrix - TP, TN, FP, FN*

```
Accuracy  = (TP+TN) / Total
Precision = TP / (TP+FP)  ← Dự đoán Pos, bao nhiêu đúng?
Recall    = TP / (TP+FN)  ← Thực tế Pos, bắt được bao nhiêu?
F1-Score  = 2×(P×R)/(P+R)
```

**Regression Metrics:**

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| MAE | mean(\|y-ŷ\|) | Sai số trung bình |
| RMSE | sqrt(mean((y-ŷ)²)) | Phạt nặng sai số lớn |
| R² | 1 - SS_res/SS_tot | % variance giải thích |

**Minh họa bằng Python:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# Huấn luyện mô hình Logistic Regression (chuẩn hóa dữ liệu trước)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Accuracy :", round(accuracy_score(y_val, y_pred), 3))
print("Precision:", round(precision_score(y_val, y_pred), 3))
print("Recall   :", round(recall_score(y_val, y_pred), 3))
print("F1-score :", round(f1_score(y_val, y_pred), 3))

# Báo cáo đầy đủ cho từng lớp
print(classification_report(y_val, y_pred, target_names=data.target_names))
```

**Regression metrics trong scikit-learn:**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([5.0, 4.2, 6.1, 3.8, 7.0])   # Giá nhà thực tế (tỷ)
y_hat  = np.array([4.8, 4.5, 5.9, 4.1, 6.4])   # Giá nhà dự đoán

print("MAE :", round(mean_absolute_error(y_true, y_hat), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_hat)), 3))
print("R²  :", round(r2_score(y_true, y_hat), 3))
```

### Step 8: Hyper-parameter Tuning

**Cross-Validation (K=5)**

![Cross Validation](images/08_cross_validation.png)

*Hình 8: 5-Fold Cross Validation - Train 5 lần, mỗi lần test trên fold khác*

**Grid Search:**
- Thử TẤT CẢ combinations
- Ví dụ: 3 params × 3 values = 27 thử nghiệm

**Regularization:**
- L1 (Lasso): Feature selection tự động
- L2 (Ridge): Giảm weights, tránh overfit

**Minh họa bằng Python:**

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-Validation 5-fold trên tập train
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print("F1 từng fold:", scores.round(3))
print("F1 trung bình:", round(scores.mean(), 3), "±", round(scores.std(), 3))

# Grid Search: thử tất cả tổ hợp tham số (C: mức regularization, penalty: L1/L2)
param_grid = {
    "logisticregression__C": [0.01, 0.1, 1, 10],
    "logisticregression__l1_ratio": [0, 1],        # 0 = L2 (Ridge), 1 = L1 (Lasso)
}
search_model = make_pipeline(StandardScaler(),
                             LogisticRegression(solver="saga", penalty="elasticnet",
                                                max_iter=5000))
grid = GridSearchCV(search_model, param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)

print("Tham số tốt nhất:", grid.best_params_)
print("F1 (CV) tốt nhất:", round(grid.best_score_, 3))
```

### Step 9: Build Pipeline

```
Raw Data
  ↓
Preprocessing (Missing, Outliers)
  ↓  
Feature Engineering (Encoding, Scaling)
  ↓
Model (Best model + Best params)
  ↓
Prediction
```

Lợi ích: Tự động hóa, tái sử dụng, dễ deploy

**Minh họa bằng Python – Pipeline hoàn chỉnh:**

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),    # Preprocessing
    ("scaler", StandardScaler()),                      # Feature Engineering
    ("model", RandomForestClassifier(n_estimators=200, random_state=42)),  # Model
])

# Huấn luyện trên train + validation, đánh giá MỘT LẦN trên test
pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
print("F1 trên TEST:", round(f1_score(y_test, pipeline.predict(X_test)), 3))

# Lưu pipeline để triển khai (deploy)
import joblib
joblib.dump(pipeline, "breast_cancer_pipeline.joblib")
# Nạp lại: pipeline = joblib.load("breast_cancer_pipeline.joblib")
```

### Step 10: Conclusion

**Checklist:**
- ☑ Performance đạt yêu cầu?
- ☑ Không overfitting?
- ☑ Có ý nghĩa business?
- ☑ Ready to deploy?

---

## 1.4 Overfitting và Underfitting

### Khái niệm và minh họa

![Overfitting vs Underfitting](images/01_overfit_underfit.png)

*Hình 9: So sánh Train Error và Test Error - Phát hiện Overfitting/Underfitting*

### So sánh

```
UNDERFITTING (Quá đơn giản)
Train Error: HIGH ↑    Test Error: HIGH ↑
→ Không học được patterns

GOOD FIT (Vừa phải)  
Train Error: LOW ↓     Test Error: LOW ↓
→ Học tốt, generalize tốt

OVERFITTING (Quá phức tạp)
Train Error: VERY LOW ↓↓   Test Error: HIGH ↑
→ Học quá kỹ, kể cả noise
```

**Minh họa bằng Python: độ sâu cây quyết định và overfitting**

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

depths = range(1, 16)
train_acc, val_acc = [], []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(tree.score(X_train, y_train))
    val_acc.append(tree.score(X_val, y_val))

plt.plot(depths, train_acc, marker="o", label="Train accuracy")
plt.plot(depths, val_acc, marker="o", label="Validation accuracy")
plt.xlabel("max_depth (độ phức tạp mô hình)")
plt.ylabel("Accuracy")
plt.title("Train vs Validation: phát hiện Overfitting")
plt.legend()
plt.show()
# Độ sâu lớn: train ≈ 100% nhưng validation không tăng (thậm chí giảm) → Overfitting
```

### Bias-Variance Tradeoff

![Bias Variance Tradeoff](images/09_bias_variance_tradeoff.png)

*Hình 10: Bias-Variance Tradeoff - Tìm điểm tối ưu*

### Cách phát hiện

```
So sánh Train vs Test:

Underfitting:  Train=60%, Test=58%  ← Cả 2 thấp
Good Fit:      Train=95%, Test=93%  ← Cả 2 cao, gần nhau  
Overfitting:   Train=99%, Test=75%  ← Chênh lệch LỚN!
```

### Cách khắc phục

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| **Underfitting** | Mô hình quá đơn giản | • Dùng model phức tạp hơn<br>• Thêm features<br>• Giảm regularization |
| **Overfitting** | Mô hình quá phức tạp | • Thêm dữ liệu<br>• Regularization (L1/L2)<br>• Cross-validation<br>• Feature selection<br>• Early stopping |

---

## 1.5 Tóm tắt

### Các loại ML và ứng dụng

| Loại | Bài toán | Ví dụ | Thuật toán phổ biến |
|------|----------|-------|---------------------|
| **Supervised - Regression** | Dự đoán số | Giá nhà, Doanh thu | Linear Regression, Random Forest, XGBoost |
| **Supervised - Classification** | Dự đoán nhãn | Spam, Chẩn đoán bệnh | Logistic Regression, SVM, Neural Networks |
| **Unsupervised - Clustering** | Phân nhóm | Phân khúc khách hàng | K-Means, Hierarchical, DBSCAN |
| **Unsupervised - Reduction** | Giảm chiều | Visualize, Nén dữ liệu | PCA, t-SNE, UMAP |

### Quy trình ML (10 bước)

```
Business Understanding:
  1. Problem Understanding
  2. Data Understanding  
  3. Feature Understanding

Data Preparation:
  4. Feature Engineering
  5. Dataset Partition
  6. Data Modelling

Model Optimization:
  7. Data Evaluation
  8. Hyper-parameter Tuning
  9. Build Pipeline
  
  10. Conclusion
```

### Các khái niệm quan trọng

✅ **Supervised**: Học từ dữ liệu có nhãn (X, y)  
✅ **Unsupervised**: Tìm cấu trúc trong dữ liệu không nhãn  
✅ **Train/Test**: Tránh overfitting  
✅ **Cross-Validation**: Đánh giá chính xác  
✅ **Metrics**: Accuracy, F1, RMSE, R²  
✅ **Overfitting**: Train tốt, Test kém → Cần regularization

### Lưu ý quan trọng

⚠️ **LUÔN** chia train/test TRƯỚC khi train  
⚠️ **KHÔNG** dùng test set để tune  
⚠️ **KIỂM TRA** overfitting/underfitting  
⚠️ **CHỌN** metric phù hợp (F1 cho imbalanced)  
⚠️ **SỬ DỤNG** cross-validation

---

## BÀI TẬP

### Bài tập 1: Phân loại bài toán

Xác định loại bài toán ML cho các tình huống sau:

1. Dự đoán giá Bitcoin ngày mai
2. Phân nhóm khách hàng theo hành vi mua hàng
3. Phát hiện giao dịch gian lận
4. Nén ảnh từ 1000 features xuống 50 features
5. Dự đoán sinh viên có tốt nghiệp đúng hạn không

**Gợi ý:** Regression, Classification, Clustering, Dimensionality Reduction

### Bài tập 2: Chọn Metric

Chọn metric phù hợp cho các bài toán:

1. Dự đoán giá nhà (sai số 500 triệu là chấp nhận được)
2. Phát hiện ung thư (bỏ sót nguy hiểm!)
3. Dự đoán khách hàng rời bỏ (99% ở lại, 1% rời)
4. So sánh 2 mô hình phân loại 10 classes

**Gợi ý:** MAE, RMSE, Accuracy, Precision, Recall, F1

### Bài tập 3: Phát hiện Overfitting

Cho kết quả 3 models:

| Model | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| A | 60% | 58% |
| B | 95% | 93% |
| C | 99% | 70% |

Hỏi: Model nào bị overfitting? Underfit? Good fit?

---

## BÀI TIẾP THEO

- **Bài 2**: Unsupervised Learning - Phân cụm (K-Means, Hierarchical, DBSCAN)
- **Bài 3**: Supervised Learning - Regression & Classification chi tiết
- **Bài 4**: Ensemble Methods (Bagging, Boosting, Stacking)
- **Bài 5**: Deep Learning cơ bản

---

**Cập nhật**: Tháng 3/2026
