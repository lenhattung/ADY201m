# BÀI 2: LẬP TRÌNH PYTHON CƠ BẢN
## Biến, Kiểu dữ liệu và Các toán tử

## 1. Biến (Variables) và Phép gán
Trong Python, biến được dùng để lưu trữ các giá trị dữ liệu. Python dùng dấu `=` để gán giá trị và **không cần khai báo kiểu** trước – kiểu dữ liệu được xác định tự động theo giá trị.

### 1.1. Toán tử gán
* **`=`**: Là toán tử gán chuẩn trong Python (Ví dụ: `x = 10`).
* **Gán nhiều biến cùng lúc:** `a, b = 10, 20`
* **Gán kết hợp phép toán:** `x += 5` (tương đương `x = x + 5`), tương tự `-=`, `*=`, `/=`.

> **Lưu ý khi chuyển từ R:** Python **không** có toán tử `<-`. Viết `x <- 10` trong Python sẽ được hiểu là phép so sánh `x < -10`!

### 1.2. Quy tắc đặt tên biến
* Tên biến có thể bao gồm chữ cái, chữ số và dấu gạch dưới `_` (**không** được dùng dấu chấm `.` như trong R).
* **Bắt đầu:** Phải bắt đầu bằng chữ cái hoặc dấu gạch dưới, **không** được bắt đầu bằng chữ số.
* **Phân biệt hoa thường:** `ketQua` và `ketqua` là hai biến khác nhau.
* **Từ khóa cấm:** Không đặt tên trùng với các từ khóa hệ thống như `if`, `else`, `for`, `True`, `False`, `None`, `def`, `class`, `import`, ...
* **Quy ước (PEP 8):** Dùng chữ thường, các từ nối bằng dấu gạch dưới (*snake_case*): `chieu_dai`, `diem_trung_binh`.

---

## 2. Các kiểu dữ liệu cơ bản (Data Types)
Python có các kiểu dữ liệu cơ bản sau:

| Kiểu dữ liệu | Tên gọi | Mô tả | Ví dụ |
| :--- | :--- | :--- | :--- |
| **float** | Số thực | Các số có phần thập phân | `10.5`, `3.14`, `2.0` |
| **int** | Số nguyên | Số nguyên, không giới hạn độ lớn (không cần hậu tố `L` như R) | `5`, `100` |
| **str** | Chuỗi | Ký tự nằm trong dấu nháy đơn hoặc kép | `"Data Science"`, `'Python'` |
| **bool** | Logic | Giá trị đúng hoặc sai (viết hoa chữ cái đầu) | `True`, `False` |
| **complex** | Số phức | Số có phần ảo, dùng hậu tố `j` | `1 + 4j` |
| **NoneType** | Rỗng | Không có giá trị | `None` |

> **Mẹo:** Sử dụng hàm `type(tên_biến)` để kiểm tra kiểu dữ liệu của biến đó.
> 
> **Khác biệt với R:** Trong R, `2` là numeric (số thực). Trong Python, `2` là `int`, còn `2.0` mới là `float`.

---

## 3. Các toán tử (Operators)

### 3.1. Toán tử số học
Dùng để thực hiện các tính toán toán học cơ bản:
* `+` : Cộng
* `-` : Trừ
* `*` : Nhân
* `/` : Chia (luôn trả về số thực: `7 / 2` → `3.5`)
* `**` : Lũy thừa (Ví dụ: `2 ** 3` kết quả là `8`). **Chú ý:** `^` trong Python là phép XOR bit, **không** phải lũy thừa!
* `%` : Chia lấy phần dư (Modulus) – tương đương `%%` trong R
* `//` : Chia lấy phần nguyên (Integer division) – tương đương `%/%` trong R

### 3.2. Toán tử so sánh
Kết quả trả về luôn là kiểu **bool** (`True` hoặc `False`):
* `==` : So sánh bằng
* `!=` : So sánh khác
* `>`  : Lớn hơn
* `<`  : Nhỏ hơn
* `>=` : Lớn hơn hoặc bằng
* `<=` : Nhỏ hơn hoặc bằng

> Python cho phép so sánh "nối chuỗi": `0 <= diem <= 10` (tương đương `diem >= 0 and diem <= 10`).

### 3.3. Toán tử logic
Dùng để kết hợp nhiều điều kiện so sánh:
* `and`  : Phép VÀ (AND) - Trả về True khi tất cả điều kiện đúng.
* `or`   : Phép HOẶC (OR) - Trả về True khi có ít nhất một điều kiện đúng.
* `not`  : Phép PHỦ ĐỊNH (NOT) - Đảo ngược giá trị logic.

> **Lưu ý quan trọng:** Với các giá trị đơn lẻ, dùng `and`, `or`, `not`. Khi làm việc với **mảng NumPy / cột pandas** (từ Bài 3 trở đi), ta phải dùng `&`, `|`, `~` và đặt mỗi điều kiện trong ngoặc: `(df["tuoi"] > 18) & (df["diem"] >= 5)`.

---

## 4. Thực hành tổng hợp
Bạn hãy copy đoạn mã sau vào Jupyter Notebook để chạy thử:

```python
# Bước 1: Khai báo các thông số
chieu_dai = 20
chieu_rong = 10
don_vi = "met"

# Bước 2: Tính toán chu vi và diện tích
chu_vi = (chieu_dai + chieu_rong) * 2
dien_tich = chieu_dai * chieu_rong

# Bước 3: Sử dụng toán tử so sánh và logic
# Kiểm tra xem diện tích có lớn hơn 150 VÀ chiều dài có lớn hơn chiều rộng không
check_dk = (dien_tich > 150) and (chieu_dai > chieu_rong)

# Bước 4: In kết quả (dùng f-string: chèn biến vào trong dấu {})
print(f"Chu vi la: {chu_vi} {don_vi}")
print(f"Dien tich la: {dien_tich} {don_vi} vuong")
print(f"Dieu kien thoa man: {check_dk}")

# Bước 5: Kiểm tra kiểu dữ liệu
print(type(don_vi))    # Trả về <class 'str'>
print(type(check_dk))  # Trả về <class 'bool'>
print(type(chu_vi))    # Trả về <class 'int'>
```

> **Về cách in kết quả:** R dùng `paste("Chu vi la:", chu_vi)`. Trong Python, cách hiện đại và dễ đọc nhất là **f-string**: `f"Chu vi la: {chu_vi}"`. Có thể định dạng số: `f"{3.14159:.2f}"` → `3.14`.
