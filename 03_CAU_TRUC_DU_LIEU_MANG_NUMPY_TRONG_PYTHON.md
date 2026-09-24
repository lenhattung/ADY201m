# BÀI 3: CẤU TRÚC DỮ LIỆU MẢNG (LIST VÀ NUMPY ARRAY) TRONG PYTHON

## 1. Mảng là gì?
Trong R, cấu trúc cơ bản nhất là **vector**. Trong Python, vai trò này được đảm nhận bởi:

* **`list`** (có sẵn trong Python): Dãy các phần tử, **có thể khác kiểu**, dùng dấu ngoặc vuông `[1, 2, 3]`.
* **`numpy.ndarray`** (thư viện NumPy): Dãy các giá trị có **CÙNG KIỂU DỮ LIỆU** (cùng `dtype`), hỗ trợ tính toán vector hóa cực nhanh. Đây là cấu trúc tương đương **vector trong R** và là nền tảng của toàn bộ hệ sinh thái Data Science trong Python.

```python
import numpy as np

ds = [1, 2, 3]            # list của Python
arr = np.array([1, 2, 3]) # mảng NumPy

print(ds * 2)   # [1, 2, 3, 1, 2, 3]  -> list: nhân = lặp lại
print(arr * 2)  # [2 4 6]             -> NumPy: nhân từng phần tử (giống R)
```

* Đặc điểm quan trọng nhất khi chuyển từ R: **Chỉ số (index) trong Python bắt đầu từ 0**, không phải từ 1.

```
Giá trị:   10   20   30   40   50
Index R:    1    2    3    4    5
Index Py:   0    1    2    3    4
Index âm:  -5   -4   -3   -2   -1   (đếm ngược từ cuối)
```

## 2. Các phương thức tạo mảng

### 2.1. Sử dụng hàm np.array() (tương đương c() trong R)
Dùng để kết hợp các phần tử đơn lẻ thành một dãy.
```python
import numpy as np

v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array(["A", "B", "C"])
print(v1, v2)
```

### 2.2. Sử dụng np.arange() (tương đương dấu hai chấm `:` trong R)
Tạo dãy số với bước nhảy mặc định là 1. **Chú ý: giá trị cuối KHÔNG được lấy.**
```python
v3 = np.arange(1, 11)        # 1 2 3 ... 10  (tương đương 1:10 trong R)
v4 = np.arange(1.5, 5.5)     # 1.5 2.5 3.5 4.5 (tương đương 1.5:4.5)
print(v3)
print(v4)
```

> Hàm `range(1, 11)` có sẵn của Python cũng tạo dãy 1..10 nhưng chỉ dùng cho số nguyên và thường dùng trong vòng lặp `for`.

### 2.3. Sử dụng np.tile() và np.repeat() (tương đương rep() trong R)
Dùng để lặp lại các phần tử.
```python
print(np.repeat(1, 10))                      # Lặp lại số 1 mười lần.
print(np.tile([1, 2], 3))                    # Kết quả: [1 2 1 2 1 2]   (rep(..., times = 3))
print(np.repeat([1, 2], 3))                  # Kết quả: [1 1 1 2 2 2]   (rep(..., each = 3))
print(np.tile(np.repeat([1, 2], 2), 2))      # Kết quả: [1 1 2 2 1 1 2 2] (times = 2, each = 2)
```

### 2.4. Sử dụng np.arange() / np.linspace() (tương đương seq() trong R)
```python
print(np.arange(1, 6))            # Tương đương seq(1, 5)
print(np.arange(1, 11, 2))        # Kết quả: [1 3 5 7 9]  (seq(1, 10, by = 2))
print(np.arange(5, 0, -1))        # Tạo dãy số lùi: [5 4 3 2 1]
print(np.linspace(0, 1, 5))       # 5 số cách đều từ 0 đến 1 (seq(0, 1, length.out = 5))

# Tạo dãy ngày tháng (dùng pandas)
import pandas as pd
print(pd.date_range("2026-01-01", periods=7, freq="D"))
```

---

## 3. Đặc điểm quan trọng: Tính đồng nhất (Coercion)
Mảng NumPy luôn chứa các phần tử cùng kiểu. Nếu trộn các kiểu dữ liệu, NumPy sẽ tự động chuyển về kiểu "mạnh" nhất theo thứ tự:
`bool < int < float < str`.

```python
v_mix = np.array([True, 1, "A"])
print(v_mix)         # ['True' '1' 'A']
print(v_mix.dtype)   # <U21  (chuỗi Unicode) vì có chứa chữ "A"

v_num = np.array([True, 1, 2.5])
print(v_num, v_num.dtype)   # [1.  1.  2.5] float64
```

> Ngược lại, `list` của Python **không** ép kiểu: `[True, 1, "A"]` vẫn giữ nguyên 3 kiểu khác nhau.

---

## 4. Thao tác và Phép toán trên mảng

### 4.1. Truy cập phần tử (Indexing & Slicing)
**Lưu ý: Index trong Python bắt đầu từ 0. Khi cắt lát `a:b`, phần tử ở vị trí `b` KHÔNG được lấy.**

```python
x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

print(x[0])              # Lấy phần tử đầu tiên (R: x[1])
print(x[-1])             # Lấy phần tử CUỐI cùng
print(np.delete(x, 1))   # Lấy tất cả trừ phần tử thứ 2 (R: x[-2])
print(x[2:5])            # Lấy phần tử thứ 3 đến thứ 5 (R: x[3:5])
print(x[[0, 4, 9]])      # Lấy phần tử thứ 1, 5 và 10 (R: x[c(1, 5, 10)])
```

> ⚠️ **Bẫy thường gặp:** Trong R, `x[-2]` nghĩa là "bỏ phần tử thứ 2". Trong Python, `x[-2]` nghĩa là "lấy phần tử **kế cuối**"!

### 4.2. Đặt tên cho phần tử (tương đương Named Vector)
Python có 2 cách phổ biến: dùng `dict` hoặc `pandas.Series` (Series giống named vector của R nhất).

```python
import pandas as pd

# Cách 1: dict
info = {"name": "Tung", "surname": "Le", "age": 18}
print(info["age"])

# Cách 2: pandas Series (có thể tính toán như vector)
vectorNamed = pd.Series(["Tung", "Le", "18"], index=["name", "surname", "age"])
print(vectorNamed)
print(vectorNamed["age"])   # Truy cập bằng tên
```

### 4.3. Các phép toán vector hóa (Vectorized)
Các phép toán (`+`, `-`, `*`, `/`, `**`) được thực hiện trên từng cặp phần tử tương ứng của hai mảng.
```python
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
print(x + y)   # Kết quả: [11 22 33]
print(x ** 2)  # Kết quả: [1 4 9]
```

### 4.4. Lọc bằng điều kiện (Boolean Indexing)
```python
x = np.array([3, 8, 1, 9, 5])
print(x > 4)          # [False  True False  True  True]
print(x[x > 4])       # [8 9 5]
print(x[(x > 2) & (x < 9)])   # Kết hợp điều kiện: dùng & và |, mỗi điều kiện đặt trong ngoặc
```

---

## 5. Các hàm thông dụng cho mảng
| R | Python (NumPy) | Ý nghĩa |
| :--- | :--- | :--- |
| `length(x)` | `len(x)` hoặc `x.size` | Độ dài mảng |
| `class(x)` | `type(x)`, `x.dtype` | Kiểu đối tượng / kiểu phần tử |
| `sum(x)`, `mean(x)` | `np.sum(x)`, `np.mean(x)` (hoặc `x.sum()`, `x.mean()`) | Tổng, trung bình |
| `max(x)`, `min(x)` | `np.max(x)`, `np.min(x)` | Lớn nhất, nhỏ nhất |
| `sort(x)` | `np.sort(x)` | Sắp xếp tăng dần |
| `rev(x)` | `x[::-1]` | Đảo ngược |

```python
x = np.array([8, 5, 9, 4, 7])
print(len(x), x.dtype)
print(np.sum(x), np.mean(x), np.max(x), np.min(x))
print(np.sort(x))
print(np.sort(x)[::-1])   # Sắp xếp giảm dần
```

---

## 6. Bài tập thực hành tại lớp

Yêu cầu:
1. Tạo một mảng `diem_thi` chứa điểm của 5 sinh viên: 8, 5, 9, 4, 7.
2. Tạo dãy số lẻ từ 1 đến 15 bằng hàm `np.arange()`.
3. Tạo một Series đặt tên (tương đương Named Vector) cho thông tin cá nhân gồm: Họ tên, Tuổi, Chuyên ngành.
4. Lọc ra các điểm thi trong mảng `diem_thi` mà có giá trị lớn hơn 5.

Gợi ý code:
```python
import numpy as np
import pandas as pd

# 1.
diem_thi = np.array([8, 5, 9, 4, 7])

# 2. (Lưu ý: giá trị cuối không được lấy nên dùng 16)
so_le = np.arange(1, 16, 2)
print(so_le)

# 3.
thong_tin = pd.Series(["Le Nhat Tung", 20, "Khoa hoc du lieu"],
                      index=["ho_ten", "tuoi", "chuyen_nganh"])
print(thong_tin)

# 4.
print(diem_thi[diem_thi > 5])
```
