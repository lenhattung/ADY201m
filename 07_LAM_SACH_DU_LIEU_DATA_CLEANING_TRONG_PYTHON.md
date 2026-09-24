# BÀI 07: LÀM SẠCH DỮ LIỆU (DATA CLEANING) TRONG PYTHON

## MỤC TIÊU HỌC TẬP

Sau bài học này, sinh viên sẽ có khả năng:
- Hiểu tầm quan trọng của việc làm sạch dữ liệu
- Xác định và xử lý các vấn đề phổ biến trong dữ liệu thô
- Áp dụng các kỹ thuật xử lý missing data
- Chuyển đổi và chuẩn hóa dữ liệu phân loại
- Sử dụng các hàm của pandas để tự động hóa quy trình làm sạch

---

## PHẦN 1: LÝ THUYẾT

### 1.1. Data Cleaning là gì?

**Data Cleaning (Làm sạch dữ liệu)** là quá trình phát hiện và sửa chữa (hoặc loại bỏ) các bản ghi không chính xác, không đầy đủ, không nhất quán hoặc không liên quan trong tập dữ liệu.

**Tại sao Data Cleaning quan trọng?**
- **"Garbage In, Garbage Out"**: Dữ liệu kém chất lượng → Kết quả phân tích sai lệch
- Chiếm **60-80% thời gian** của một dự án Data Science
- Ảnh hưởng trực tiếp đến độ chính xác của mô hình Machine Learning
- Giúp đưa ra quyết định kinh doanh chính xác hơn

---

### 1.2. Các vấn đề phổ biến trong dữ liệu thô

#### **1.2.1. Missing Data (Dữ liệu thiếu)**

**Nguyên nhân:**
- Lỗi nhập liệu
- Thiết bị đo lường hỏng
- Người dùng không muốn cung cấp thông tin
- Lỗi trong quá trình thu thập dữ liệu

**Các phương pháp xử lý:**

| Phương pháp | Mô tả | Ưu điểm | Nhược điểm |
|------------|-------|---------|------------|
| **Loại bỏ** | Xóa các dòng/cột có giá trị thiếu (`dropna`) | Đơn giản, nhanh | Mất thông tin, giảm kích thước dataset |
| **Điền giá trị trung bình/trung vị** | Thay NaN bằng mean/median (`fillna`) | Giữ được kích thước dữ liệu | Có thể làm giảm độ biến thiên |
| **Điền giá trị phổ biến nhất (Mode)** | Dùng cho biến phân loại | Phù hợp với categorical data | Có thể tạo bias |
| **Forward Fill / Backward Fill** | Dùng giá trị trước/sau đó (`ffill`, `bfill`) | Tốt cho time series | Không phù hợp với dữ liệu ngẫu nhiên |
| **Imputation nâng cao** | Dùng ML (`KNNImputer`, `IterativeImputer` của scikit-learn) | Chính xác hơn | Phức tạp, tốn thời gian |

**Quy tắc chung:**
- Nếu < 5% dữ liệu thiếu → có thể loại bỏ
- Nếu 5-20% → cân nhắc điền giá trị
- Nếu > 20% → cần phân tích nguyên nhân sâu hơn

---

#### **1.2.2. Dữ liệu không nhất quán (Inconsistent Data)**

**Ví dụ:**
- Giới tính: "M", "Male", "Nam", "male" → cùng ý nghĩa nhưng khác format
- Ngày tháng: "01/12/2024", "2024-12-01", "Dec 1, 2024"
- Đơn vị: 100 cm vs 1 m

**Giải pháp:**
- Chuẩn hóa (standardization): đưa về cùng một định dạng (`.str.lower()`, `.replace()`, `pd.to_datetime()`)
- Sử dụng kiểu **category** trong pandas để quản lý categorical data

---

#### **1.2.3. Dữ liệu trùng lặp (Duplicate Data)**

**Nguyên nhân:**
- Nhập liệu nhiều lần
- Ghép nối nhiều nguồn dữ liệu
- Lỗi hệ thống

**Xử lý:**
```python
import pandas as pd

data = pd.DataFrame({"ID": [1, 2, 2, 3], "Score": [8, 9, 9, 7]})

# Tìm dòng trùng lặp
print(data.duplicated())

# Loại bỏ trùng lặp
data = data.drop_duplicates()
# hoặc
data = data[~data.duplicated()]
print(data)
```

---

#### **1.2.4. Outliers (Giá trị ngoại lai)**

**Định nghĩa:** Giá trị cực đoan, khác biệt rõ rệt so với phần lớn dữ liệu

**Phát hiện:**
- Biểu đồ boxplot
- Z-score (giá trị chuẩn hóa)
- IQR (Interquartile Range)

**Xử lý:**
- Kiểm tra xem có phải lỗi nhập liệu không
- Nếu là lỗi → sửa hoặc loại bỏ
- Nếu là giá trị thật → giữ lại hoặc transform (`np.log`, `np.sqrt`)

---

### 1.3. Làm việc với Categorical Data trong pandas

#### **Kiểu category trong pandas**

**Category** là kiểu dữ liệu đặc biệt trong pandas để lưu trữ categorical data (tương đương `factor` trong R).

**Các loại category:**

1. **Nominal (Không có thứ tự)**
   - Ví dụ: giới tính, màu sắc, tên thành phố
   ```python
   sex = pd.Categorical(["Male", "Female", "Male"])
   ```

2. **Ordinal (Có thứ tự)**
   - Ví dụ: học lực (Yếu < Trung bình < Khá < Giỏi)
   ```python
   grade = pd.Categorical(["Good", "Bad", "Average"],
                          categories=["Bad", "Average", "Good"],
                          ordered=True)
   ```

**Tại sao dùng category?**
- Tiết kiệm bộ nhớ (lưu dưới dạng số nguyên)
- Dễ dàng thống kê và phân tích
- Dễ tạo biến dummy cho machine learning (`pd.get_dummies`)
- Kiểm soát được các giá trị hợp lệ

---

### 1.4. Quy trình Data Cleaning chuẩn

```
1. Hiểu dữ liệu (Data Understanding)
   ↓
2. Khám phá dữ liệu (Exploratory Data Analysis)
   ↓
3. Xử lý Missing Data
   ↓
4. Xử lý Outliers
   ↓
5. Chuẩn hóa & Transform Data
   ↓
6. Xử lý dữ liệu trùng lặp
   ↓
7. Kiểm tra và validate kết quả
   ↓
8. Lưu dữ liệu sạch
```

---

## PHẦN 2: THỰC HÀNH VỚI DATASET THỰC TẾ

### 2.1. Giới thiệu Dataset

**Tên dataset:** Student Alcohol Consumption  
**Nguồn:** UCI Machine Learning Repository  
**Mô tả:** Dữ liệu về tình trạng tiêu thụ rượu của học sinh trung học ở Bồ Đào Nha

**Các biến quan trọng:**
- `school`: Trường học (GP hoặc MS)
- `sex`: Giới tính
- `age`: Tuổi
- `Medu`, `Fedu`: Trình độ học vấn của cha mẹ (0-4)
- `studytime`: Thời gian học mỗi tuần
- `Dalc`: Mức độ uống rượu trong tuần (1-5)
- `Walc`: Mức độ uống rượu cuối tuần (1-5)

---

### 2.2. BƯỚC 1: Load và Khám phá Dữ liệu

```python
import numpy as np
import pandas as pd

# Load dữ liệu
alcohol = pd.read_csv("data/dataset - student alcohol consumption/student-alcohol.csv")

# Xem 5 dòng đầu tiên
print(alcohol.head())

# Kiểm tra cấu trúc dữ liệu
alcohol.info()

# Tóm tắt thống kê
print(alcohol.describe(include="all"))
```

**Giải thích:**
- `head()`: Xem nhanh dữ liệu có gì
- `info()`: Kiểm tra kiểu dữ liệu và số giá trị không thiếu của từng cột
- `describe()`: Thống kê mô tả (count, mean, std, min, max, các tứ phân vị, ...)

```python
# Loại bỏ cột đầu tiên (có thể là cột ID không cần thiết)
print(alcohol.iloc[:, 1:].head())   # Xem trước khi xóa
alcohol = alcohol.iloc[:, 1:]       # Xóa cột 1
```

**Lưu ý:** 
- `.iloc[:, 1:]` nghĩa là "lấy tất cả các dòng, lấy từ cột thứ 2 trở đi" (tương đương `[,-1]` trong R)
- Cách khác: `alcohol.drop(columns=alcohol.columns[0])`
- Luôn kiểm tra trước khi xóa để tránh mất dữ liệu quan trọng

---

### 2.3. BƯỚC 2: Xử lý Missing Data

#### **2.3.1. Phát hiện Missing Data**

```python
# Tìm các dòng có dữ liệu thiếu
print(alcohol[alcohol.isna().any(axis=1)])

# Đếm số dòng bị thiếu
print(len(alcohol[alcohol.isna().any(axis=1)]))

# Đếm số ô thiếu theo từng cột
print(alcohol.isna().sum()[alcohol.isna().sum() > 0])
```

**Giải thích:**
- `isna()`: trả về True tại những ô bị thiếu
- `.any(axis=1)`: True nếu dòng có ít nhất một ô thiếu (tương đương `!complete.cases()` trong R)
- Kết quả cho biết chính xác dòng nào, cột nào bị thiếu

---

#### **2.3.2. Xử lý biến số (Numeric) - Tuổi (age)**

```python
# Kiểm tra phân bố tuổi
print(alcohol["age"].describe())

# Tính trung vị (pandas tự động bỏ qua NaN – không cần na.rm = TRUE)
print(alcohol["age"].median())

# Điền missing values bằng median
alcohol["age"] = alcohol["age"].fillna(alcohol["age"].median())

# Kiểm tra xem còn NaN không
print(alcohol["age"].isna().sum())
```

**Tại sao dùng median thay vì mean?**
- **Median** ít bị ảnh hưởng bởi outliers hơn mean
- Ví dụ: Tuổi = [15, 16, 16, 17, 100] → mean ≈ 33, median = 16
- Với tuổi học sinh, median hợp lý hơn

---

#### **2.3.3. Xử lý biến phân loại (Categorical) - Mjob**

```python
# Kiểm tra lại missing data
print(alcohol[alcohol.isna().any(axis=1)])

# Điền giá trị "other" cho dòng thứ 63 (Mjob bị thiếu)
# Dòng thứ 63 trong R ứng với nhãn index 62 trong pandas (index bắt đầu từ 0)
alcohol.loc[62, "Mjob"] = "other"

# Cách tổng quát hơn: điền "other" cho MỌI ô Mjob bị thiếu
alcohol["Mjob"] = alcohol["Mjob"].fillna("other")

# Kiểm tra lại
print(alcohol[alcohol.isna().any(axis=1)])
```

**Lý do chọn "other":**
- Biến Mjob (nghề nghiệp mẹ) có các giá trị: teacher, health, services, at_home, other
- Khi không biết → gán "other" hợp lý hơn là xóa cả dòng

---

### 2.4. BƯỚC 3: Chuyển đổi Categorical Data thành kiểu category

Trong R ta dùng `factor(x, levels, labels)`. Trong pandas, việc **đổi nhãn** và **tạo category** thường làm theo 2 bước:
1. `.map({giá_trị_cũ: nhãn_mới})` để đổi nhãn
2. `pd.Categorical(..., categories=[...], ordered=...)` để tạo kiểu category

Để viết gọn, ta tạo một hàm tiện ích dùng lại cho nhiều cột:

```python
def to_factor(series, levels, labels=None, ordered=False):
    """Mô phỏng hàm factor(x, levels, labels, ordered) của R."""
    if labels is None:
        labels = levels
    mapped = series.map(dict(zip(levels, labels)))
    return pd.Categorical(mapped, categories=labels, ordered=ordered)
```

#### **2.4.1. Biến nhị phân đơn giản (Binary Variables)**

```python
# School - Trường học
print(alcohol["school"].value_counts())

alcohol["school"] = to_factor(alcohol["school"],
                              levels=["GP", "MS"],
                              labels=["Gabriel Pereira", "Mousinho da Silveira"])
```

**Giải thích:**
- `levels`: Các giá trị hiện tại trong dữ liệu ("GP", "MS")
- `labels`: Nhãn mới dễ hiểu hơn (tên đầy đủ của trường)

---

```python
# Sex - Giới tính
print(alcohol["sex"].value_counts())

alcohol["sex"] = to_factor(alcohol["sex"],
                           levels=["F", "M"],
                           labels=["female", "male"])
```

---

```python
# Address - Nơi ở
print(alcohol["address"].value_counts())

alcohol["address"] = to_factor(alcohol["address"],
                               levels=["R", "U"],
                               labels=["rural", "urban"])
```

---

```python
# Family size - Quy mô gia đình
print(alcohol["famsize"].value_counts())

alcohol["famsize"] = to_factor(alcohol["famsize"],
                               levels=["GT3", "LE3"],
                               labels=["more than 3", "less or equal to 3"])
```

**GT3 = Greater Than 3, LE3 = Less or Equal to 3**

---

```python
# Parent's cohabitation status - Tình trạng chung sống của cha mẹ
print(alcohol["Pstatus"].value_counts())

alcohol["Pstatus"] = to_factor(alcohol["Pstatus"],
                               levels=["A", "T"],
                               labels=["living apart", "living together"])
```

---

#### **2.4.2. Ordinal Categories (Biến có thứ tự)**

```python
# Mother's education - Trình độ học vấn của mẹ
print(alcohol["Medu"].value_counts().sort_index())

alcohol["Medu"] = to_factor(alcohol["Medu"],
                            levels=[0, 1, 2, 3, 4],
                            labels=["none", "primary", "primary higher",
                                    "secondary", "higher"],
                            ordered=True)
```

**Quan trọng:**
- `ordered=True`: Đánh dấu đây là biến có thứ tự
- none < primary < primary higher < secondary < higher
- pandas cho phép so sánh (`<`, `>`), lấy `min()`, `max()` và sắp xếp theo đúng thứ tự này

---

```python
# Father's education - Trình độ học vấn của cha
print(alcohol["Fedu"].value_counts().sort_index())

alcohol["Fedu"] = to_factor(alcohol["Fedu"],
                            levels=[0, 1, 2, 3, 4],
                            labels=["none", "primary", "primary higher",
                                    "secondary", "higher"],
                            ordered=True)
```

---

```python
# Reason to choose this school - Lý do chọn trường
print(alcohol["reason"].value_counts())

alcohol["reason"] = alcohol["reason"].astype("category")
```

**Lưu ý:** Khi không chỉ định categories, pandas tự động lấy các giá trị unique (sắp theo alphabet) làm categories

---

```python
# Kiểm tra cấu trúc sau khi chuyển đổi
alcohol.info()
print(alcohol.describe(include="category"))
```

---

```python
# Guardian - Người giám hộ
print(alcohol["guardian"].value_counts())
alcohol["guardian"] = alcohol["guardian"].astype("category")
```

---

#### **2.4.3. Biến thời gian và khoảng cách**

```python
# Travel time - Thời gian đi học
print(alcohol["traveltime"].describe())
print(alcohol["traveltime"].value_counts().sort_index())

# Ý nghĩa: 1 - <15 phút, 2 - 15-30 phút, 3 - 30-60 phút, 4 - >1 giờ
alcohol["traveltime"] = to_factor(alcohol["traveltime"],
                                  levels=[1, 2, 3, 4],
                                  labels=["0-15 min", "15-30 min",
                                          "30-60 min", "above 60 min"],
                                  ordered=True)
```

---

```python
# Study time - Thời gian học mỗi tuần
print(alcohol["studytime"].describe())
print(alcohol["studytime"].value_counts().sort_index())

# Ý nghĩa: 1 - <2 giờ, 2 - 2-5 giờ, 3 - 5-10 giờ, 4 - >10 giờ
alcohol["studytime"] = to_factor(alcohol["studytime"],
                                 levels=[1, 2, 3, 4],
                                 labels=["0-2 hours", "2-5 hours",
                                         "5-10 hours", "above 10 hours"],
                                 ordered=True)
```

---

### 2.5. BƯỚC 4: Tự động hóa với vòng lặp và apply()

Thay vì viết code lặp đi lặp lại cho nhiều biến tương tự, ta dùng **vòng lặp `for`**, **dict comprehension** hoặc **`.apply()`** (tương đương `lapply()` trong R).

#### **2.5.1. Xử lý nhóm biến Yes/No**

```python
# School support - Hỗ trợ từ trường
print(alcohol["schoolsup"].value_counts())
alcohol["schoolsup"] = pd.Categorical(alcohol["schoolsup"], categories=["no", "yes"])
```

**Nhận xét:** Có 8 biến cùng dạng yes/no → lặp code 8 lần rất tốn công!

---

```python
# Liệt kê tất cả biến binary (yes/no)
binaryVariables = ["schoolsup", "famsup", "paid", "activities",
                   "nursery", "higher", "internet", "romantic"]

# Xem dữ liệu
print(alcohol[binaryVariables].head())

# Kiểm tra từng biến: áp dụng value_counts cho từng cột
for col in binaryVariables:
    print(col, alcohol[col].value_counts(dropna=False).to_dict())
```

**Giải thích:**
- Vòng lặp `for col in danh_sách:` áp dụng cùng một thao tác lên từng cột
- Cách viết gọn tương đương `lapply` của R: `{col: alcohol[col].value_counts() for col in binaryVariables}` (dict comprehension)

---

#### **2.5.2. Xử lý dữ liệu bẩn - Case study: biến internet**

```python
# Phát hiện vấn đề
print(alcohol["internet"].value_counts())
```

**Vấn đề:** Biến `internet` có nhiều giá trị (0, 1, NO, YES, no, yes) → lỗi nhập liệu!

**Giải pháp: Chuẩn hóa**

```python
# Đảm bảo cột là chuỗi (số 0/1 khi đọc từ CSV có thể là kiểu số)
alcohol["internet"] = alcohol["internet"].astype(str)

# Chuyển "0" → "no"
alcohol.loc[alcohol["internet"] == "0", "internet"] = "no"

# Chuyển "1" → "yes"
alcohol.loc[alcohol["internet"] == "1", "internet"] = "yes"

# Chuyển "NO" → "no"
alcohol.loc[alcohol["internet"] == "NO", "internet"] = "no"

# Chuyển "YES" → "yes"
alcohol.loc[alcohol["internet"] == "YES", "internet"] = "yes"

# Kiểm tra lại
print(alcohol["internet"].value_counts())
```

**Cách viết gọn hơn** (làm tất cả trong một dòng):
```python
alcohol["internet"] = (alcohol["internet"].astype(str)
                       .str.lower()                           # "NO" → "no", "YES" → "yes"
                       .replace({"0": "no", "1": "yes"}))     # 0 → no, 1 → yes
print(alcohol["internet"].value_counts())
```

**Bài học:** Trong thực tế, dữ liệu thường không "sạch" → cần kiểm tra kỹ!

---

```python
# Kiểm tra lại tất cả biến binary
for col in binaryVariables:
    print(col, alcohol[col].value_counts().to_dict())

# Chuyển đổi tất cả cùng lúc
alcohol[binaryVariables] = alcohol[binaryVariables].apply(
    lambda x: pd.Categorical(x, categories=["no", "yes"])
)
print(alcohol[binaryVariables].dtypes)
```

**Kết quả:** Thay vì viết 8 lần code giống nhau → chỉ cần 1 lệnh!

---

### 2.6. BƯỚC 5: Xử lý biến mức độ (1-5 scale)

```python
# Tiếp tục kiểm tra
alcohol.info()

# Các biến có mức độ từ 1-5
leveledVariables = ["freetime", "goout", "Dalc", "Walc"]

# Xem phân bố
print(alcohol[leveledVariables].describe())
```

**Ý nghĩa:**
- `freetime`: Thời gian rảnh sau giờ học
- `goout`: Tần suất đi chơi với bạn bè
- `Dalc`: Mức độ uống rượu trong ngày thường (Daily Alcohol Consumption)
- `Walc`: Mức độ uống rượu cuối tuần (Weekend Alcohol Consumption)

**Scale:** 1 = rất thấp, 2 = thấp, 3 = trung bình, 4 = cao, 5 = rất cao

---

```python
# Chuyển đổi
alcohol[leveledVariables] = alcohol[leveledVariables].apply(
    lambda x: to_factor(x,
                        levels=[1, 2, 3, 4, 5],
                        labels=["very low", "low", "average", "high", "very high"],
                        ordered=True)
)
```

---

```python
# Health status - Tình trạng sức khỏe
print(alcohol["health"].describe())
print(alcohol["health"].value_counts().sort_index())

# Ý nghĩa: 1 = rất tệ, 2 = tệ, 3 = trung bình, 4 = tốt, 5 = rất tốt
alcohol["health"] = to_factor(alcohol["health"],
                              levels=[1, 2, 3, 4, 5],
                              labels=["very bad", "bad", "average",
                                      "good", "very good"],
                              ordered=True)
```

---

### 2.7. BƯỚC 6: Kiểm tra kết quả cuối cùng

```python
# Xem cấu trúc dữ liệu sau khi clean
alcohol.info()

# Tóm tắt thống kê
print(alcohol.describe(include="all"))

# Kiểm tra missing data
print(alcohol.isna().any(axis=1).sum())  # Phải = 0
```

**Checklist:**
- Không còn missing data
- Tất cả categorical variables đã là kiểu `category`
- Các biến có thứ tự đã được đánh dấu `ordered=True`
- Nhãn dữ liệu rõ ràng, dễ hiểu

> **Lưu ý khi lưu file:** File CSV không lưu được kiểu `category`. Muốn giữ nguyên kiểu dữ liệu, lưu bằng định dạng **Parquet** (`df.to_parquet(...)`) hoặc **pickle** (`df.to_pickle(...)`).

---

## PHẦN 3: TÓM TẮT VÀ BÀI TẬP

### 3.1. Tóm tắt các hàm quan trọng

| pandas | Công dụng | Ví dụ | R tương đương |
|-----|-----------|-------|------|
| `df.info()` | Xem cấu trúc dữ liệu | `data.info()` | `str()` |
| `df.describe()` | Tóm tắt thống kê | `data.describe()` | `summary()` |
| `df.head()` | Xem n dòng đầu | `data.head(10)` | `head()` |
| `df.dropna()` | Bỏ các dòng có NaN | `data.dropna()` | `data[complete.cases(data),]` |
| `isna()` | Kiểm tra giá trị NaN | `data["age"].isna()` | `is.na()` |
| `fillna()` | Điền giá trị thiếu | `data["age"].fillna(0)` | `x[is.na(x)] <- 0` |
| `pd.Categorical()` | Tạo categorical variable | `pd.Categorical(x, categories, ordered)` | `factor()` |
| `value_counts()` | Đếm tần số | `data["sex"].value_counts()` | `table()` |
| `apply()` | Áp dụng hàm lên từng cột | `data[cols].apply(func)` | `lapply()` |

---

### 3.2. Best Practices

**NÊN:**
1. **Luôn backup dữ liệu gốc** trước khi clean (`data_backup = data.copy()`)
2. **Kiểm tra từng bước** bằng `describe()`, `info()`, `head()`
3. **Tự động hóa** các tác vụ lặp lại với vòng lặp, `apply()`, hàm tự viết
4. **Comment code** để giải thích lý do làm gì
5. **Validate kết quả** sau khi clean

**KHÔNG NÊN:**
1. Xóa dữ liệu mà không hiểu nguyên nhân
2. Điền missing values một cách tùy tiện
3. Bỏ qua outliers mà không kiểm tra
4. Lưu đè lên file gốc
5. Làm việc trên toàn bộ dataset lớn ngay từ đầu (nên test trên subset nhỏ trước: `data.sample(1000)`)

> ⚠️ **Bẫy riêng của Python:** `data_backup = data` **không** tạo bản sao – hai biến cùng trỏ tới một DataFrame, sửa biến này thì biến kia cũng đổi theo. Luôn dùng `data.copy()`.

---

### 3.3. Quy trình làm việc được khuyến nghị

```python
# 1. Load dữ liệu
data = pd.read_csv("file.csv")

# 2. Backup (bắt buộc dùng .copy())
data_backup = data.copy()

# 3. Khám phá
data.info()
print(data.describe(include="all"))
print(data.head())

# 4. Xử lý missing data
# ... code xử lý ...

# 5. Xử lý categorical data
# ... code xử lý ...

# 6. Validate
data.info()
print(data.describe(include="all"))
print(data.isna().any(axis=1).sum())

# 7. Lưu dữ liệu sạch
data.to_csv("data_cleaned.csv", index=False)
```

---

### 3.4. Bài tập thực hành

**Bài 1:** Sử dụng dataset `iris`
```python
import statsmodels.api as sm
iris = sm.datasets.get_rdataset("iris").data
```
Thực hiện:
1. Tạo thêm 10 giá trị NaN ngẫu nhiên trong cột `Sepal.Length` (gợi ý: `np.random.default_rng().choice(len(iris), 10, replace=False)`)
2. Điền missing values bằng median
3. Chuyển cột `Species` thành category có thứ tự (theo thứ tự alphabet)

---

**Bài 2:** Tạo dataset giả lập có vấn đề:
```python
messy_data = pd.DataFrame({
    "ID": [1, 2, 3, 4, 5, 2],                          # Có ID trùng lặp
    "Age": [25, np.nan, 35, 150, 28, 25],              # Có NaN và outlier
    "Gender": ["M", "F", "male", "Female", "m", "F"],  # Không nhất quán
    "Score": [85, 90, 88, np.nan, 92, 85],
})
```

Yêu cầu:
1. Loại bỏ dòng trùng lặp
2. Xử lý missing values
3. Xử lý outlier (Age = 150)
4. Chuẩn hóa cột Gender thành category với các mức ("Male", "Female")

---

**Bài 3:** (Nâng cao) Viết function tự động clean data:
```python
def auto_clean(data):
    """Tự động:
    1. Loại bỏ dòng trùng lặp
    2. Điền missing values (numeric: median, categorical: mode)
    3. Chuyển cột chuỗi thành category
    Trả về: DataFrame đã clean
    """
    ...
```

---

## TÀI LIỆU THAM KHẢO

1. **Sách:**
   - "Python for Data Analysis" (3rd ed.) - Wes McKinney (tác giả thư viện pandas)
   - "Python Data Science Handbook" - Jake VanderPlas

2. **Online:**
   - [pandas Documentation – Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
   - [pandas Documentation – Categorical data](https://pandas.pydata.org/docs/user_guide/categorical.html)

3. **Datasets để thực hành:**
   - UCI Machine Learning Repository
   - Kaggle Datasets
   - `sklearn.datasets`, `seaborn.load_dataset()`, `statsmodels.api.datasets.get_rdataset()`

---

## KẾT LUẬN

Data Cleaning là kỹ năng quan trọng nhất của một Data Scientist. Một dataset sạch không chỉ giúp phân tích chính xác hơn mà còn tiết kiệm thời gian và công sức trong các bước sau.

**Nhớ:** 
> "Data cleaning is not a one-time task, it's a continuous process."

**Câu hỏi?** 
Liên hệ: [Email/Office Hours]

---

*Bài giảng được biên soạn bởi: [Tên giảng viên]*  
*Cập nhật lần cuối: [Ngày/tháng/năm]*
