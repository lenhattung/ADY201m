# BÀI 4: KIỂU DỮ LIỆU PHÂN LOẠI (CATEGORICAL) TRONG PANDAS

## 1. Tại sao cần Categorical khi đã có mảng chuỗi?
Thông thường, mảng chuỗi (`str`) dùng để lưu trữ văn bản tự do. Tuy nhiên, trong phân tích dữ liệu, chúng ta thường gặp loại dữ liệu "định danh" hoặc "thứ bậc".

* Ví dụ: Nếu dùng mảng chuỗi để lưu size áo "S", "M", "L", Python sẽ hiểu đây chỉ là các chữ cái và sắp xếp chúng theo bảng chữ cái (L -> M -> S). 
* Để Python hiểu được "S" nhỏ hơn "M" và nhỏ hơn "L", chúng ta phải dùng kiểu **Categorical** của thư viện pandas (tương đương **factor** trong R).

Categorical giúp:
- Tiết kiệm bộ nhớ (lưu dưới dạng các con số nguyên – gọi là *codes* – đi kèm nhãn).
- Xác định đúng thứ tự ưu tiên trong phân tích và vẽ biểu đồ.
- Hỗ trợ các mô hình thống kê phân loại (tạo biến giả – dummy variables).

---

## 2. Categorical là gì?
Categorical là kiểu dữ liệu đặc biệt dùng để lưu trữ dữ liệu phân loại (categorical data) với các mức độ xác định. Trong pandas, các mức độ được gọi là **categories** (tương đương `levels` trong R).

| R | pandas |
| :--- | :--- |
| `factor(x)` | `pd.Categorical(x)` hoặc `pd.Series(x, dtype="category")` |
| `levels(x)` | `x.categories` (hoặc `s.cat.categories` với Series) |
| `ordered = TRUE` | `ordered=True` |
| `table(x)`, `summary(x)` | `s.value_counts()` |
| `as.integer(x)` | `x.codes` (**bắt đầu từ 0**) |

---

## 3. Phân loại Categorical

### 3.1. Unordered Categorical (Không có thứ tự)
Dùng cho dữ liệu chỉ mang tính chất định danh, không so sánh cao thấp.
Ví dụ: Màu sắc, giới tính, quốc gia...

Mã mẫu:
```python
import pandas as pd

colors = pd.Categorical(["red", "green", "blue", "yellow"])
print(type(colors))
print(colors.categories)   # Các mức được sắp theo alphabet
print(colors.codes)        # Mã số nguyên bên trong
```

### 3.2. Ordered Categorical (Có thứ tự)
Dùng cho dữ liệu có sự phân cấp rõ rệt.
Ví dụ: Trình độ học vấn, mức độ hài lòng, size quần áo...

Mã mẫu:
```python
ratings = pd.Categorical(["low", "high", "medium"],
                         categories=["low", "medium", "high"],  # Xác định thứ tự
                         ordered=True)
print(ratings)
print(ratings < "high")    # So sánh được vì có thứ tự
print(ratings.min(), ratings.max())
```

---

## 4. Các ví dụ thực tế trong phân tích dữ liệu

### Ví dụ 1: Quản lý trình độ giáo dục (Education)
```python
education = pd.Series(["High School", "Bachelor", "Master", "PhD"],
                      dtype=pd.CategoricalDtype(
                          categories=["High School", "Bachelor", "Master", "PhD"],
                          ordered=True))
print(education)
print(education.cat.categories)
```

### Ví dụ 2: Phân loại điểm số học sinh
```python
grades = pd.Series(
    pd.Categorical(["Giỏi", "Khá", "Trung bình", "Giỏi", "Khá", "Yếu"],
                   categories=["Yếu", "Trung bình", "Khá", "Giỏi"],
                   ordered=True)
)
# sort=False: giữ đúng thứ tự các mức thay vì sắp theo tần số
print(grades.value_counts(sort=False))
```

### Ví dụ 3: Khảo sát mức độ hài lòng (Customer Satisfaction)
```python
import matplotlib.pyplot as plt

satisfaction = pd.Series(
    pd.Categorical(["Rất thích", "Thích", "Bình thường", "Không thích", "Thích", "Rất thích"],
                   categories=["Không thích", "Bình thường", "Thích", "Rất thích"],
                   ordered=True)
)
tab_sat = satisfaction.value_counts(sort=False)

tab_sat.plot(kind="bar", color="lightgreen", rot=0)
plt.title("Kết quả khảo sát khách hàng")
plt.show()
```
---

## 5. Lưu ý khi chọn kiểu dữ liệu
- Số thực/Số nguyên (Tuổi, ID): `int` / `float`
- Văn bản tự do (Tên, địa chỉ): `str` (trong pandas là `object` hoặc `string`)
- Dữ liệu thể loại (Size, Học vấn): `category`

> Kiểm tra kiểu của từng cột trong bảng dữ liệu bằng `df.dtypes` hoặc `df.info()`.

---

## 6. Bài tập thực hành tại lớp
Yêu cầu:
1. Tạo một mảng chứa kích cỡ áo của 10 khách hàng: M, L, S, XL, M, S, M, L, XL, M.
2. Chuyển mảng trên thành Categorical có thứ tự từ nhỏ đến lớn: S < M < L < XL.
3. Sử dụng hàm `value_counts()` để thống kê số lượng từng size áo.

Mã gợi ý:
```python
import pandas as pd
import matplotlib.pyplot as plt

sizes = pd.Series(
    pd.Categorical(["M", "L", "S", "XL", "M", "S", "M", "L", "XL", "M"],
                   categories=["S", "M", "L", "XL"],
                   ordered=True)
)
thong_ke = sizes.value_counts(sort=False)
print(thong_ke)

thong_ke.plot(kind="bar", color="orange", rot=0)
plt.title("Thống kê kích cỡ áo")
plt.show()
```
