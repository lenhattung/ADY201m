# BÀI 5: CẤU TRÚC DỮ LIỆU NHIỀU CHIỀU: MA TRẬN (NUMPY 2D) VÀ DICT/LIST

## 1. Đặt vấn đề: Tại sao mảng 1 chiều là chưa đủ?

Trong các bài trước, chúng ta đã làm việc với **mảng 1 chiều** (dãy số hoặc chuỗi). Tuy nhiên, dữ liệu thực tế trong Khoa học Dữ liệu thường phức tạp hơn:

1. **Dữ liệu dạng bảng:** Bảng điểm sinh viên, bảng doanh thu theo tháng, dữ liệu ảnh (pixel). Chúng có hàng và cột → Cần **Ma trận (NumPy 2D array)**.
2. **Dữ liệu hỗn hợp:** Một hồ sơ bệnh án bao gồm: Tên (chữ), Tuổi (số), Lịch sử khám (bảng), Kết quả xét nghiệm (True/False). Mảng hay Ma trận không thể chứa lẫn lộn các kiểu này → Cần **dict** (từ điển) hoặc **list**.

| R | Python |
| :--- | :--- |
| `matrix` | `numpy.ndarray` 2 chiều (hoặc `pandas.DataFrame` khi cần tên hàng/cột) |
| `list` có tên (`list(a = 1, b = 2)`) | `dict` (`{"a": 1, "b": 2}`) |
| `list` không tên | `list` (`[1, "a", True]`) |

---

## 2. Ma trận (Matrix)

### 2.1. Ma trận trong thực tế phân tích dữ liệu

Ma trận là cấu trúc dữ liệu **2 chiều** (hàng và cột).

* **Đặc điểm cốt lõi:** Tính **đồng nhất (Homogeneous)**. Tất cả phần tử bắt buộc phải cùng kiểu (tất cả là số, hoặc tất cả là chữ).
* **Ứng dụng:**
    * Xử lý ảnh (ảnh đen trắng là một ma trận các con số từ 0-255).
    * Đại số tuyến tính trong Machine Learning (nhân ma trận trọng số: `A @ B`).
    * Lưu trữ dữ liệu dạng bảng đơn giản.

### 2.2. Khởi tạo Ma trận

Cú pháp: `np.array(...)` hoặc `np.arange(...).reshape(so_hang, so_cot, order=...)`

**Ví dụ 1: Tạo ma trận số học cơ bản**

```python
import numpy as np

# Điền số theo hàng (MẶC ĐỊNH của NumPy, order="C")
mat_row = np.arange(1, 7).reshape(2, 3)
print(mat_row)

# Điền số theo cột (giống MẶC ĐỊNH của R, order="F")
mat_col = np.arange(1, 7).reshape(2, 3, order="F")
print(mat_col)

# Tạo trực tiếp từ list lồng nhau (mỗi list con là một hàng)
mat = np.array([[1, 2, 3],
                [4, 5, 6]])
print(mat.shape)   # (2, 3): 2 hàng, 3 cột  (R: dim())
```

> ⚠️ **Khác biệt với R:** R mặc định điền **theo cột** (`byrow = FALSE`), còn NumPy mặc định điền **theo hàng**.

**Ví dụ 2: Ứng dụng quản lý điểm số (Đặt tên cho hàng/cột)**
Mảng NumPy **không** có tên hàng/cột. Khi cần đặt tên (như `rownames`, `colnames` trong R), ta dùng `pandas.DataFrame` – giúp truy xuất dữ liệu minh bạch hơn là chỉ dùng số thứ tự.

```python
import pandas as pd

# Dữ liệu điểm của 3 sinh viên cho 4 môn học
diem = np.array([
    [8.5, 9.0, 7.5, 8.8],   # Điểm SV 1
    [9.2, 7.8, 8.5, 9.0],   # Điểm SV 2
    [7.6, 8.2, 8.9, 8.5],   # Điểm SV 3
])

# Gán nhãn để dữ liệu có ý nghĩa
grades = pd.DataFrame(diem,
                      index=["Le Nhat Tung", "Nguyen Van A", "Le Thi C"],
                      columns=["Toán", "Văn", "Anh", "Tin"])
print(grades)
```

### 2.3. Truy cập và trích xuất dữ liệu (Indexing)

Tư duy truy cập: `[hàng, cột]` – **chỉ số bắt đầu từ 0**.
* Với mảng NumPy: `mat[i, j]`
* Với DataFrame: `.iloc[i, j]` (theo **vị trí**) hoặc `.loc["tên hàng", "tên cột"]` (theo **tên**)

```python
# 1. Lấy điểm môn Văn (cột 2) của Nguyen Van A (hàng 2)
print(diem[1, 1])                          # NumPy
print(grades.loc["Nguyen Van A", "Văn"])   # pandas theo tên

# 2. Lấy toàn bộ bảng điểm của Le Nhat Tung (Hàng 1)
print(grades.iloc[0, :])

# 3. Lấy điểm môn Tin học của cả lớp (Cột 4)
# Cách dùng chỉ số
print(grades.iloc[:, 3])
# Cách dùng tên cột (trực quan hơn)
print(grades["Tin"])

# 4. Lấy điểm Toán và Văn của 2 sinh viên đầu tiên (Slicing)
print(grades.iloc[0:2][["Toán", "Văn"]])
```

### 2.4. Các phép toán thống kê trên Ma trận

Trong Data Science, chúng ta thường cần tính toán tổng hợp theo chiều dọc (theo biến số) hoặc chiều ngang (theo đối tượng). Trong NumPy/pandas, điều này được điều khiển bằng tham số **`axis`**:
* `axis=0`: tính **theo cột** (dọc xuống) – tương đương `colMeans`, `colSums`
* `axis=1`: tính **theo hàng** (ngang qua) – tương đương `rowMeans`, `rowSums`

```python
import matplotlib.pyplot as plt

# Tính điểm trung bình từng môn học (Theo cột)
print(grades.mean(axis=0))

# Tính điểm tổng kết của từng sinh viên (Theo hàng)
print(grades.mean(axis=1))

# Vẽ biểu đồ nhanh so sánh điểm trung bình các môn
grades.mean(axis=0).plot(kind="bar", color="lightblue", rot=0)
plt.title("Điểm trung bình các môn")
plt.show()
```

---

## 3. Dict và List – "Chiếc balo" vạn năng

### 3.1. Dict - "Chiếc balo" có ngăn đặt tên

Nếu mảng/ma trận là chiếc hộp chỉ đựng được một loại đồ vật, thì **dict** là chiếc balo có nhiều ngăn được **đặt tên** (key), mỗi ngăn chứa được mọi thứ: số, chuỗi, mảng, DataFrame, và thậm chí là một dict khác. Đây là cấu trúc tương đương **list có tên** trong R.

* **Ứng dụng thực tế:**
    * Kết quả trả về của các mô hình, cấu hình tham số (hyper-parameters) của mô hình học máy.
    * Lưu trữ cấu trúc JSON từ API (web scraping) – JSON được đọc vào Python chính là dict.
    * Hồ sơ cá nhân tổng hợp.

### 3.2. Khởi tạo Dict

```python
# Một hồ sơ sinh viên điển hình
student_profile = {
    "info": ["Nguyen Van A", "K15_CNTT"],             # List chuỗi
    "age": 20,                                        # Số nguyên
    "grades": np.array([8.5, 9.0, 7.5]),              # Mảng số thực
    "passed": True,                                   # Logic
    "details": np.arange(1, 5).reshape(2, 2),         # Thậm chí chứa cả Ma trận
}
```

### 3.3. Truy cập dữ liệu trong Dict và List (Rất quan trọng)

Đây là phần sinh viên hay nhầm lẫn nhất khi chuyển từ R sang Python.

| Thao tác | R (list) | Python |
| :--- | :--- | :--- |
| Lấy nội dung theo tên | `x$grades` hoặc `x[["grades"]]` | `x["grades"]` (dict) |
| Lấy nội dung theo vị trí | `x[[3]]` | `ds[2]` (list) – **index từ 0** |
| Lấy "list con" | `x[1:2]` | `ds[0:2]` (list) |

```python
# Cách 1: Truy cập bằng key (Khuyên dùng vì dễ đọc)
print(student_profile["grades"])

# Cách 2: Lấy giá trị thực sự để tính toán
avg_score = np.mean(student_profile["grades"])
print(avg_score)

# Cách 3: Thêm dữ liệu mới vào Dict
student_profile["email"] = "vana@university.edu.vn"
print(list(student_profile.keys()))   # Kiểm tra các thành phần (R: names())

# Dùng .get() để tránh lỗi khi key không tồn tại
print(student_profile.get("phone", "Chưa có số điện thoại"))
```

### 3.4. List – danh sách theo thứ tự (không đặt tên)

```python
# List có thể chứa các kiểu khác nhau
hon_hop = ["Nguyen Van A", 20, [8.5, 9.0, 7.5], True]

print(hon_hop[0])        # Phần tử đầu tiên (nội dung)
print(hon_hop[2][1])     # Phần tử thứ 2 của list con: 9.0
print(hon_hop[0:2])      # List con gồm 2 phần tử đầu

hon_hop.append("vana@university.edu.vn")   # Thêm phần tử vào cuối
print(len(hon_hop))
```

---

## 4. Tổng kết so sánh (Cheat Sheet)

| Tiêu chí | Mảng 1D (`np.array`) | Ma trận (`np.array` 2D / DataFrame) | Dict / List |
| --- | --- | --- | --- |
| **Số chiều** | 1 chiều | 2 chiều (Hàng, Cột) | Linh hoạt, lồng nhau |
| **Kiểu dữ liệu** | Đồng nhất (Chỉ 1 loại) | Đồng nhất (NumPy) | **Hỗn hợp** (Mọi loại) |
| **Truy cập** | `x[i]` | `m[i, j]`, `.iloc[i, j]`, `.loc[...]` | `d["key"]`, `ds[i]` |
| **Ví dụ** | Dãy điểm số | Bảng điểm, Ảnh số | Kết quả Model AI, JSON |

---

## 5. Bài tập thực hành (Lab)

### Bài 1: Phân tích doanh số (Ma trận)

Tạo một ma trận `sales` lưu doanh số của 3 sản phẩm (iPhone, Samsung, Xiaomi) trong 3 tháng đầu năm (T1, T2, T3).

1. Tạo ma trận với dữ liệu tùy ý (ví dụ: từ 100 đến 500).
2. Đặt tên hàng là tên sản phẩm, tên cột là các tháng (dùng `pd.DataFrame`).
3. Tính tổng doanh thu của từng sản phẩm trong cả quý (`.sum(axis=1)`).
4. Tính doanh thu trung bình của từng tháng (`.mean(axis=0)`).

### Bài 2: Quản lý nhân sự (Dict)

Tạo một dict tên `employee` chứa thông tin của một giảng viên:

1. `name`: Tên giảng viên.
2. `subjects`: List chứa tên 3 môn dạy (ví dụ: "Python", "R", "ML").
3. `schedule`: Ma trận 2x3 thể hiện số tiết dạy (2 ngày, 3 ca).
4. **Yêu cầu:** Dùng lệnh truy cập để in ra môn học thứ 2 mà giảng viên này dạy (gợi ý: `employee["subjects"][1]`).

---
