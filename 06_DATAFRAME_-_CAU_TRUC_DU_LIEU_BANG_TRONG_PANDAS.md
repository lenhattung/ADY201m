# BÀI 6: DATAFRAME - CẤU TRÚC DỮ LIỆU BẢNG TRONG PANDAS

## 1. Giới thiệu về DataFrame
Nếu mảng NumPy là nền tảng, ma trận là bảng tính đồng nhất, thì **DataFrame** của thư viện **pandas** chính là "trái tim" của phân tích dữ liệu trong Python.

* **Định nghĩa:** DataFrame là một tập hợp các cột (mỗi cột là một `pandas.Series`) có cùng độ dài, dùng chung một **chỉ mục hàng (index)**.
* **Đặc điểm:**
    * Cấu trúc dạng bảng (hàng và cột).
    * **Quan trọng:** Các cột khác nhau có thể chứa các kiểu dữ liệu khác nhau (Ví dụ: Cột 1 là Tên (`str`), Cột 2 là Tuổi (`int`), Cột 3 là Đã tốt nghiệp (`bool`)).
    * Tương đương với một "Sheet" trong Excel, một bảng trong SQL, hoặc `data.frame` trong R.

---

## 2. Tạo và Kiểm tra DataFrame

### 2.1. Tạo DataFrame thủ công
Cách phổ biến nhất là truyền vào một **dict**: mỗi key là tên cột, mỗi value là dữ liệu của cột đó.

```python
import pandas as pd

# Tạo các cột thành phần
column1 = [1, 2, 3]                  # Số nguyên
column2 = ["Tung", "Tom", "Anna"]    # Chuỗi
column3 = [True, True, False]        # Logic

# Ghép thành DataFrame
dataset1 = pd.DataFrame({"column1": column1,
                         "column2": column2,
                         "column3": column3})

# Xem kết quả
print(dataset1)
```

> Cột bên trái (0, 1, 2) là **index** – chỉ mục hàng, mặc định bắt đầu từ 0.

### 2.2. Đổi tên cột (Column Names)

```python
# Đổi tên tất cả các cột
dataset1.columns = ["ID", "Name", "Passed"]

# Đổi tên một cột cụ thể (khuyên dùng rename)
dataset1 = dataset1.rename(columns={"Name": "StudentName"})
print(dataset1)
```

### 2.3. Kiểm tra dữ liệu (Rất quan trọng)

Trước khi phân tích, luôn phải kiểm tra cấu trúc dữ liệu.

| R | pandas | Ý nghĩa |
| :--- | :--- | :--- |
| `head(df)` | `df.head()` | Xem vài dòng đầu |
| `tail(df)` | `df.tail()` | Xem vài dòng cuối |
| `str(df)` | `df.info()` / `df.dtypes` | Cấu trúc, kiểu dữ liệu từng cột |
| `summary(df)` | `df.describe()` | Tóm tắt thống kê (min, max, mean...) |
| `dim(df)` | `df.shape` | Kích thước (Số hàng, Số cột) |

```python
print(dataset1.head())    # Xem vài dòng đầu
print(dataset1.tail())    # Xem vài dòng cuối
dataset1.info()           # Xem cấu trúc (kiểu dữ liệu từng cột)
print(dataset1.describe(include="all"))  # Tóm tắt thống kê cho mọi cột
print(dataset1.shape)     # Kích thước (Số hàng, Số cột)
```

---

## 3. Truy xuất và Thao tác dữ liệu

### 3.1. Truy xuất (Indexing)

Có 4 cách chính để lấy dữ liệu từ DataFrame:

1. **Theo vị trí `[hàng, cột]` – dùng `.iloc` (chỉ số bắt đầu từ 0):**
```python
print(dataset1.iloc[0, 1])   # Lấy dòng 1, cột 2 (R: dataset1[1, 2])
```

2. **Theo tên cột (trả về Series) - Dùng `df["tên"]` hoặc `df.tên`:** *(Phổ biến nhất – tương đương `$` trong R)*
```python
print(dataset1["StudentName"])
print(dataset1.StudentName)   # Chỉ dùng được khi tên cột không có dấu cách/ký tự đặc biệt
```

3. **Theo tên cột (trả về DataFrame con) - Dùng `df[["tên"]]` (hai lớp ngoặc):**
```python
print(dataset1[["StudentName"]])
```

4. **Theo nhãn hàng và tên cột – dùng `.loc`:**
```python
print(dataset1.loc[0, "StudentName"])            # Hàng có index 0, cột StudentName
print(dataset1.loc[:, ["ID", "StudentName"]])    # Tất cả hàng, 2 cột
```

### 3.2. Mở rộng DataFrame

* **Thêm cột mới:**
```python
# Cách 1: Gán trực tiếp (Khuyên dùng)
dataset1["Score"] = [8.5, 9.0, 7.5]

# Cách 2: Dùng assign (trả về DataFrame mới)
dataset1 = dataset1.assign(Location=["HN", "HCM", "DN"])
print(dataset1)
```

* **Thêm dòng mới (`pd.concat` – tương đương `rbind`):**
*Lưu ý: Dòng mới nên có cùng tên cột tương ứng.*
```python
newRow = pd.DataFrame({"ID": [4], "StudentName": ["Duong"], "Passed": [True],
                       "Score": [8.0], "Location": ["Hue"]})
dataset1 = pd.concat([dataset1, newRow], ignore_index=True)
print(dataset1)
```

---

## 4. Các thao tác nâng cao (Data Wrangling)

### 4.1. Sắp xếp dữ liệu (Sorting)

Sử dụng phương thức `sort_values()` (tương đương `df[order(...), ]` trong R).

```python
# Sắp xếp theo ID tăng dần
print(dataset1.sort_values("ID"))

# Sắp xếp theo Score giảm dần
print(dataset1.sort_values("Score", ascending=False))
```

### 4.2. Lọc dữ liệu (Filtering)

Lọc ra các dòng thỏa mãn điều kiện.

```python
# Cách 1: Dùng chỉ số logic (Boolean Indexing)
# Lấy sinh viên đã đậu (Passed == True)
passed_students = dataset1[dataset1["Passed"] == True]

# Kết hợp nhiều điều kiện: dùng & (và), | (hoặc), mỗi điều kiện đặt trong ngoặc
good_students = dataset1[(dataset1["Passed"] == True) & (dataset1["Score"] > 8)]

# Cách 2: Dùng query() (Dễ đọc hơn – tương đương subset() trong R)
good_students = dataset1.query("Passed == True and Score > 8")
print(good_students)
```

### 4.3. Gộp bảng dữ liệu (Merging / Joins)

Tương tự như SQL, pandas dùng hàm `pd.merge()` để nối hai bảng dựa trên một cột khóa (Key).

Giả sử ta có 2 bảng:

* `set1`: Chứa thông tin sản phẩm (Key: IdClient)
* `set2`: Chứa thông tin vùng miền (Key: IdClient)

```python
set1 = pd.DataFrame({"IdClient": [1, 2, 3, 4],
                     "Product": ["Laptop", "Phone", "Tablet", "Watch"]})
set2 = pd.DataFrame({"IdClient": [2, 3, 4, 5],
                     "Region": ["Bắc", "Trung", "Nam", "Nam"]})

# 1. Inner Join (Chỉ lấy phần chung có ở cả 2 bảng)
print(pd.merge(set1, set2, on="IdClient"))

# 2. Outer Join (Lấy tất cả, thiếu điền NaN) – R: all = TRUE
print(pd.merge(set1, set2, on="IdClient", how="outer"))

# 3. Left Join (Giữ nguyên bảng bên trái set1) – R: all.x = TRUE
print(pd.merge(set1, set2, on="IdClient", how="left"))
```

---

## 5. Làm sạch dữ liệu (Data Cleaning)

Dữ liệu thực tế thường bị khuyết thiếu. Trong pandas, giá trị thiếu được biểu diễn là **`NaN`** (hoặc `None`, `pd.NA`) – tương đương `NA` trong R.

### 5.1. Phát hiện dữ liệu thiếu

```python
import numpy as np

dataset1.loc[1, "Score"] = np.nan    # Tạo một ô bị thiếu để minh họa

print(dataset1.isna())               # Kiểm tra toàn bộ bảng
print(dataset1.isna().sum())         # Đếm số ô thiếu theo từng cột
print(dataset1.isna().sum().sum())   # Đếm tổng số ô bị thiếu
```

### 5.2. Xử lý dữ liệu thiếu

1. **Xóa dòng thiếu (`dropna` – tương đương `complete.cases`):**
```python
# Chỉ giữ lại các dòng đầy đủ dữ liệu
clean_data = dataset1.dropna()
print(clean_data)
```

2. **Điền dữ liệu thiếu (Imputation):**
```python
# Điền NaN bằng giá trị trung bình (pandas tự bỏ qua NaN khi tính mean)
mean_val = dataset1["Score"].mean()
dataset1["Score"] = dataset1["Score"].fillna(mean_val)
print(dataset1)
```

---

## 6. Bài tập thực hành (Lab)

> **Nạp dữ liệu mẫu có sẵn của R vào Python:** Các bộ dữ liệu kinh điển `iris`, `mtcars`, `CO2`... có thể tải qua thư viện `statsmodels` (cần kết nối Internet):
> ```python
> import statsmodels.api as sm
> iris = sm.datasets.get_rdataset("iris").data
> ```

### Bài tập 1: Thao tác cơ bản với `iris`

1. Hiển thị các dòng từ 40 đến 120, nhưng chỉ lấy các dòng chia hết cho 3 (bước nhảy = 3).
2. **Gợi ý:** Dùng slicing `start:stop:step` với `.iloc`.

**Giải:**

```python
import statsmodels.api as sm
iris = sm.datasets.get_rdataset("iris").data

# Dòng thứ 40 trong R ứng với vị trí 39 trong Python; stop=120 để lấy tới dòng thứ 120
print(iris.iloc[39:120:3])
```

### Bài tập 2: Phân tích bộ dữ liệu `CO2`

1. Lọc ra các cây có nguồn gốc (`Type`) là "Quebec" và điều trị (`Treatment`) là "chilled".
2. Tìm các mẫu có `uptake` > 40, sau đó sắp xếp kết quả theo nồng độ `conc` tăng dần.

**Giải:**

```python
CO2 = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/CO2.csv",
                  index_col=0)

# Câu 1
cau1 = CO2.query("Type == 'Quebec' and Treatment == 'chilled'")
print(cau1)

# Câu 2 (Kết hợp filter và sort)
temp = CO2[CO2["uptake"] > 40]
ketqua = temp.sort_values("conc")
# Hoặc viết gộp 1 dòng (method chaining):
ketqua = CO2[CO2["uptake"] > 40].sort_values("conc")
print(ketqua)
```

### Bài tập 3: Xử lý dữ liệu thiếu (Advanced)

Cho bộ dữ liệu `missCO2` (được tạo từ `CO2` với các giá trị thiếu ngẫu nhiên và cột `weight` dạng chữ như `"30kg"`).

```python
# Tạo bộ dữ liệu missCO2 để thực hành
rng = np.random.default_rng(42)
missCO2 = CO2.copy().reset_index(drop=True)
missCO2.loc[rng.choice(len(missCO2), 8, replace=False), "uptake"] = np.nan
missCO2.loc[rng.choice(len(missCO2), 5, replace=False), "conc"] = np.nan
missCO2["weight"] = [f"{w}kg" for w in rng.integers(20, 40, len(missCO2))]
print(missCO2.head())
```

1. Tìm các dòng có ít nhất 1 giá trị thiếu.
2. Điền các giá trị `uptake` bị thiếu bằng số 20.
3. Trích xuất số từ cột `weight` (ví dụ "30kg" -> 30) và lưu vào cột mới.

**Giải:**

```python
# 1. Tìm dòng thiếu (dòng có ít nhất 1 ô NaN)
print(missCO2[missCO2.isna().any(axis=1)])

# 2. Điền giá trị 20
missCO2["uptake"] = missCO2["uptake"].fillna(20)

# 3. Xử lý chuỗi (Text processing)
# Dùng .str.replace để thay chữ "kg" bằng rỗng, sau đó chuyển sang số
missCO2["weightNumber"] = pd.to_numeric(missCO2["weight"].str.replace("kg", ""))
print(missCO2.head())
```

---

## 7. Ghi nhớ (Key Takeaways)

1. **DataFrame** (pandas) là cấu trúc quan trọng nhất cho Machine Learning và Thống kê trong Python.
2. Dùng `df["tên_cột"]` để truy cập cột nhanh chóng; `.iloc` theo vị trí, `.loc` theo nhãn.
3. Dùng `sort_values()` để sắp xếp, `query()` hoặc `df[điều_kiện]` để lọc.
4. Luôn xử lý giá trị thiếu (`NaN`) trước khi chạy mô hình phân tích.
