# BÀI 8: XÂY DỰNG FUNCTION TRONG PYTHON

## Mục tiêu học tập
- Hiểu khái niệm và vai trò của function trong lập trình Python
- Nắm vững cú pháp khai báo và định nghĩa function
- Phân biệt các loại tham số và giá trị trả về
- Xây dựng được các function đơn giản và phức tạp
- Áp dụng function vào giải quyết bài toán thực tế

---

## 3.1 Giới thiệu về Function

### 3.1.1 Function là gì?

Function (hàm) là một khối mã được đặt tên, có thể được tái sử dụng nhiều lần trong chương trình. Function giúp:
- **Tổ chức code**: Chia nhỏ chương trình thành các phần có chức năng rõ ràng
- **Tái sử dụng**: Viết một lần, sử dụng nhiều lần
- **Dễ bảo trì**: Sửa lỗi hoặc cải tiến chỉ cần sửa ở một chỗ
- **Dễ đọc**: Code rõ ràng, dễ hiểu hơn

### 3.1.2 Phân loại Function trong Python

Python có hai loại function chính:
1. **Built-in Functions**: Hàm có sẵn trong Python hoặc trong thư viện
   - `print()`, `len()`, `sum()`, `max()`, `np.mean()`, ...
   
2. **User-defined Functions**: Hàm do người dùng tự định nghĩa bằng từ khóa `def`
   - Tạo ra để giải quyết bài toán cụ thể

```python
import numpy as np

# Ví dụ Built-in function
numbers = [1, 2, 3, 4, 5]
print(np.mean(numbers))  # 3.0
print(sum(numbers))      # 15
print(len(numbers))      # 5

# Ví dụ User-defined function (sẽ học sau)
def calculate_average(x):
    return sum(x) / len(x)

print(calculate_average(numbers))  # 3.0
```

---

## 3.2 Cú pháp khai báo Function

### 3.2.1 Cấu trúc cơ bản

```python
def function_name(parameter1, parameter2):
    # Thân hàm (function body) – THỤT LỀ 4 dấu cách
    # Các câu lệnh xử lý
    result = parameter1 + parameter2
    return result  # Giá trị trả về (optional)
```

**Các thành phần:**
- `def`: Từ khóa khai báo hàm (R dùng `function_name <- function(...) { }`)
- `function_name`: Tên của hàm (nên đặt tên có ý nghĩa, dạng *snake_case*)
- `parameter1, parameter2`: Các tham số đầu vào (có thể có hoặc không)
- Dấu hai chấm `:` và **thụt lề** thay cho cặp ngoặc nhọn `{ }` của R
- `return`: Trả về kết quả (không bắt buộc)

> ⚠️ **Khác biệt quan trọng với R:** Python dùng **thụt lề (indentation)** để xác định khối lệnh. Thụt lề sai sẽ gây lỗi `IndentationError` hoặc làm code chạy sai logic.

### 3.2.2 Ví dụ đơn giản

```python
# Function không có tham số
def say_hello():
    print("Xin chào!")

say_hello()  # Xin chào!

# Function có một tham số
def say_hello_to(name):
    message = f"Xin chào {name}!"
    print(message)

say_hello_to("Tùng")  # Xin chào Tùng!

# Function có nhiều tham số
def calculate_sum(a, b):
    result = a + b
    return result

print(calculate_sum(5, 3))  # 8
```

---

## 3.3 Tham số (Parameters)

### 3.3.1 Tham số bắt buộc

```python
# Tất cả tham số đều bắt buộc
def divide(a, b):
    return a / b

print(divide(10, 2))   # 5.0
# divide(10)           # Lỗi TypeError! Thiếu tham số b
```

### 3.3.2 Tham số mặc định (Default Parameters)

```python
# Tham số có giá trị mặc định
def greet(name, greeting="Xin chào"):
    message = f"{greeting} {name}!"
    return message

print(greet("Tùng"))                    # Xin chào Tùng!
print(greet("Tùng", "Chào buổi sáng"))  # Chào buổi sáng Tùng!

# Ví dụ thực tế: Tính lũy thừa
def power(base, exponent=2):
    return base ** exponent

print(power(5))      # 25 (5^2)
print(power(5, 3))   # 125 (5^3)
print(power(2, 10))  # 1024 (2^10)
```

> **Quy tắc:** Tham số có giá trị mặc định phải đứng **sau** tham số bắt buộc.

### 3.3.3 Truyền tham số theo tên (Keyword Arguments)

```python
# Truyền tham số theo thứ tự
def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)
    return bmi

print(round(calculate_bmi(70, 1.75), 2))  # 22.86

# Truyền tham số theo tên (không cần đúng thứ tự)
print(round(calculate_bmi(height=1.75, weight=70), 2))  # 22.86

# Kết hợp (tham số theo vị trí phải đứng trước)
print(round(calculate_bmi(70, height=1.75), 2))  # 22.86
```

### 3.3.4 Tham số không giới hạn (*args và **kwargs)

Tương đương `...` trong R:
- `*args`: nhận nhiều tham số theo vị trí → gom thành **tuple**
- `**kwargs`: nhận nhiều tham số theo tên → gom thành **dict**

```python
# Sử dụng *args để nhận số lượng tham số không xác định
def calculate_average(*args):
    avg = sum(args) / len(args)
    return avg

print(calculate_average(1, 2, 3))         # 2.0
print(calculate_average(10, 20, 30, 40))  # 25.0
print(calculate_average(5))               # 5.0

# Ví dụ kết hợp với tham số cố định
def print_info(title, *values):
    print(title, ":")
    for val in values:
        print("-", val)

print_info("Danh sách sinh viên", "Tùng", "Hùng", "Dũng")
# Output:
# Danh sách sinh viên :
# - Tùng
# - Hùng
# - Dũng

# **kwargs: nhận tham số có tên
def print_profile(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

print_profile(ten="Tùng", tuoi=30, nganh="Khoa học dữ liệu")
```

---

## 3.4 Giá trị trả về (Return Value)

### 3.4.1 Sử dụng return

```python
# Trả về giá trị rõ ràng
def square(x):
    result = x * x
    return result

print(square(5))  # 25
```

### 3.4.2 Không có return → trả về None

> ⚠️ **Khác biệt với R:** R tự động trả về biểu thức cuối cùng. **Python thì KHÔNG** – nếu không có `return`, hàm trả về `None`.

```python
def square_no_return(x):
    x * x          # Tính xong nhưng không trả về!

print(square_no_return(5))  # None

# Ví dụ phức tạp hơn: dùng if/elif/else với return
def get_grade(score):
    if score >= 85:
        return "Xuất sắc"
    elif score >= 70:
        return "Giỏi"
    elif score >= 55:
        return "Khá"
    elif score >= 40:
        return "Trung bình"
    else:
        return "Yếu"

print(get_grade(88))  # Xuất sắc
print(get_grade(65))  # Khá
```

### 3.4.3 Trả về nhiều giá trị

```python
import statistics

# Cách 1: Trả về dict (tương đương list có tên trong R)
def calculate_stats(numbers):
    result = {
        "mean": np.mean(numbers),
        "median": np.median(numbers),
        "sd": np.std(numbers, ddof=1),   # ddof=1: độ lệch chuẩn MẪU (giống sd() của R)
        "min": min(numbers),
        "max": max(numbers),
    }
    return result

scores = [75, 82, 68, 91, 77, 85, 73]
stats = calculate_stats(scores)

print(round(stats["mean"], 2))  # 78.71
print(stats["median"])          # 77.0
print(round(stats["sd"], 2))    # 7.89
print(stats["min"])             # 68
print(stats["max"])             # 91

# Cách 2: Trả về tuple và "giải nén" (unpacking) – rất phổ biến trong Python
def get_min_max(x):
    return min(x), max(x)

lo, hi = get_min_max([3, 7, 2, 9, 5])
print(lo, hi)   # 2 9
```

> ⚠️ **Lưu ý:** `np.std(x)` mặc định chia cho **n** (độ lệch chuẩn tổng thể). Để giống `sd()` của R (chia cho **n − 1**), phải dùng `np.std(x, ddof=1)`. Hàm `statistics.stdev(x)` và `pandas.Series.std()` mặc định đã chia cho n − 1.

### 3.4.4 Return sớm

```python
import math

# Sử dụng return để thoát sớm
def check_positive(x):
    if x <= 0:
        return "Số không dương"

    # Code chỉ chạy khi x > 0
    result = math.sqrt(x)
    return f"Căn bậc hai: {result}"

print(check_positive(-5))  # Số không dương
print(check_positive(16))  # Căn bậc hai: 4.0
```

---

## 3.5 Phạm vi biến (Variable Scope)

### 3.5.1 Biến cục bộ (Local Variables)

```python
# Biến trong function chỉ tồn tại trong function
def test_function():
    local_var = "Tôi là biến cục bộ"
    print(local_var)

test_function()     # Tôi là biến cục bộ
# print(local_var)  # Lỗi NameError! local_var không tồn tại bên ngoài
```

### 3.5.2 Biến toàn cục (Global Variables)

```python
# Biến toàn cục
global_var = 100

def use_global():
    print(global_var)  # Có thể đọc biến toàn cục

use_global()  # 100

# Sửa đổi biến toàn cục (không khuyến khích)
def modify_global():
    global global_var   # Khai báo dùng biến toàn cục (tương đương <<- trong R)
    global_var = 200

print(global_var)    # 100
modify_global()
print(global_var)    # 200
```

### 3.5.3 Ví dụ về phạm vi biến

```python
x = 10  # Biến toàn cục

def demo_scope():
    x = 20  # Biến cục bộ (khác với biến toàn cục)
    print(f"Trong function: {x}")

demo_scope()                    # Trong function: 20
print(f"Ngoài function: {x}")   # Ngoài function: 10

# x toàn cục không bị thay đổi
```

> ⚠️ **Bẫy với dữ liệu "mutable":** List, dict, mảng NumPy, DataFrame được truyền **theo tham chiếu**. Nếu hàm sửa trực tiếp (ví dụ `lst.append(...)`, `df["col"] = ...`), đối tượng bên ngoài **cũng bị thay đổi** – điều không xảy ra trong R. Muốn an toàn, hãy tạo bản sao bên trong hàm: `df = df.copy()`.

---

## 3.6 Các ví dụ Function thực tế

### 3.6.1 Function tính toán thống kê

```python
# Tính hệ số biến thiên (Coefficient of Variation)
def cv(x):
    x = np.asarray(x, dtype=float)
    mean_val = np.nanmean(x)           # nanmean: bỏ qua NaN (giống na.rm = TRUE)
    sd_val = np.nanstd(x, ddof=1)
    cv_percent = (sd_val / mean_val) * 100
    return cv_percent

scores = [75, 80, 85, 90, 95]
print(round(cv(scores), 2))  # 9.3 (%)

# Tính IQR và tìm outliers
def find_outliers(x):
    x = np.asarray(x)
    Q1 = np.quantile(x, 0.25)
    Q3 = np.quantile(x, 0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = x[(x < lower_bound) | (x > upper_bound)]

    result = {
        "outliers": outliers,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "n_outliers": len(outliers),
    }
    return result

data = [23, 25, 27, 29, 31, 33, 35, 100]  # 100 là outlier
result = find_outliers(data)
print(result)
```

### 3.6.2 Function xử lý dữ liệu

```python
# Chuẩn hóa dữ liệu (Normalization – Min-Max)
def normalize(x):
    x = np.asarray(x, dtype=float)
    min_val = np.nanmin(x)
    max_val = np.nanmax(x)
    normalized = (x - min_val) / (max_val - min_val)
    return normalized

data = [10, 20, 30, 40, 50]
print(normalize(data))  # [0.   0.25 0.5  0.75 1.  ]

# Chuẩn hóa Z-score
def standardize(x):
    x = np.asarray(x, dtype=float)
    mean_val = np.nanmean(x)
    sd_val = np.nanstd(x, ddof=1)
    z_scores = (x - mean_val) / sd_val
    return z_scores

scores = [60, 70, 80, 90, 100]
print(np.round(standardize(scores), 4))
# [-1.2649 -0.6325  0.      0.6325  1.2649]
```

### 3.6.3 Function xử lý chuỗi

```python
# Viết hoa chữ cái đầu
def capitalize_first(text):
    first_char = text[0]      # Ký tự đầu (R: substr(text, 1, 1))
    rest_chars = text[1:]     # Phần còn lại
    result = first_char.upper() + rest_chars.lower()
    return result

print(capitalize_first("xin chào"))  # Xin chào
# Python có sẵn phương thức: "xin chào".capitalize()

# Đếm số từ trong câu
def count_words(sentence):
    words = sentence.split()   # split() không tham số tự loại bỏ khoảng trắng thừa
    return len(words)

print(count_words("Xin chào   các bạn sinh viên"))  # 6
```

### 3.6.4 Function tính toán toán học

```python
# Tính giai thừa
def factorial_custom(n):
    if n < 0:
        return "Không tính được giai thừa của số âm"
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):   # range(2, n+1) = 2, 3, ..., n
        result = result * i
    return result

print(factorial_custom(5))  # 120

# Kiểm tra số nguyên tố
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Chỉ cần kiểm tra các ước lẻ từ 3 đến căn bậc hai của n
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

print(is_prime(17))  # True
print(is_prime(15))  # False
print(is_prime(3))   # True

# Tính số Fibonacci thứ n (quy ước: F1 = 0, F2 = 1)
def fibonacci(n):
    if n == 1:
        return 0
    if n == 2:
        return 1

    fib = [0] * n        # Tạo list n phần tử 0 (R: numeric(n))
    fib[0] = 0
    fib[1] = 1
    for i in range(2, n):
        fib[i] = fib[i - 1] + fib[i - 2]

    return fib[n - 1]

# Dãy 10 số Fibonacci đầu tiên (R: sapply(1:10, fibonacci))
print([fibonacci(i) for i in range(1, 11)])
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

> **Chú ý khi chuyển vòng lặp từ R:** Trong R, `for (i in 3:n)` khi `n < 3` sẽ chạy **lùi** (3, 2, ...) – nguồn gốc của nhiều lỗi khó phát hiện. Trong Python, `range(3, n)` khi `n <= 3` đơn giản là **không chạy lần nào**.

### 3.6.5 Function xử lý điều kiện và validation

```python
# Kiểm tra độ tuổi hợp lệ
def validate_age(age):
    if not isinstance(age, (int, float)):
        return "Tuổi phải là số"
    if age < 0:
        return "Tuổi không thể âm"
    if age > 150:
        return "Tuổi không hợp lý"
    return "Hợp lệ"

print(validate_age(25))     # Hợp lệ
print(validate_age(-5))     # Tuổi không thể âm
print(validate_age("abc"))  # Tuổi phải là số

# Phân loại BMI
def classify_bmi(weight, height):
    # Validation
    if weight <= 0 or height <= 0:
        return "Cân nặng và chiều cao phải dương"

    bmi = weight / (height ** 2)

    if bmi < 18.5:
        category = "Thiếu cân"
    elif bmi < 25:
        category = "Bình thường"
    elif bmi < 30:
        category = "Thừa cân"
    else:
        category = "Béo phì"

    result = {
        "bmi": round(bmi, 2),
        "category": category,
    }
    return result

print(classify_bmi(70, 1.75))
# {'bmi': 22.86, 'category': 'Bình thường'}
```

---

## 3.7 Function lồng nhau và Function làm tham số

### 3.7.1 Function lồng nhau

```python
# Định nghĩa function bên trong function khác
def outer_function(x):
    def inner_function(y):
        return y * 2

    result = inner_function(x) + 10
    return result

print(outer_function(5))  # 20 (5*2 + 10)
```

### 3.7.2 Function làm tham số

```python
# Truyền function như một tham số
def apply_operation(x, y, operation):
    return operation(x, y)

# Định nghĩa các operation
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def power(a, b):
    return a ** b

# Sử dụng
print(apply_operation(5, 3, add))       # 8
print(apply_operation(5, 3, multiply))  # 15
print(apply_operation(5, 3, power))     # 125

# Ví dụ với hàm ẩn danh (lambda – tương đương function(a, b) {...} của R)
print(apply_operation(10, 2, lambda a, b: a - b))  # 8
```

### 3.7.3 Ứng dụng thực tế: thay thế "apply family" của R

| R | Python |
| :--- | :--- |
| `lapply(list, f)` | `{k: f(v) for k, v in d.items()}` hoặc `[f(v) for v in ds]` |
| `sapply(1:10, f)` | `[f(i) for i in range(1, 11)]` hoặc `list(map(f, range(1, 11)))` |
| `apply(df, 2, f)` | `df.apply(f)` (theo cột) |
| `apply(df, 1, f)` | `df.apply(f, axis=1)` (theo hàng) |

```python
# Sử dụng function với dict comprehension (tương đương lapply)
numbers = {"a": range(1, 6), "b": range(6, 11), "c": range(11, 16)}

# Tính trung bình cho mỗi phần tử
print({k: np.mean(v) for k, v in numbers.items()})
# {'a': 3.0, 'b': 8.0, 'c': 13.0}

# Sử dụng function tự định nghĩa
def calculate_range(x):
    return max(x) - min(x)

print({k: calculate_range(v) for k, v in numbers.items()})
# {'a': 4, 'b': 4, 'c': 4}

# Tương đương sapply: trả về pandas Series có tên
import pandas as pd
print(pd.Series({k: sum(v) for k, v in numbers.items()}))
# a    15
# b    40
# c    65
```

---

## 3.8 Best Practices khi viết Function

### 3.8.1 Đặt tên function

```python
# ❌ TÊN KHÔNG TỐT
def f(x):
    return x * 2

def func1(x, y):
    return x + y

# ✅ TÊN TỐT (động từ + danh từ, snake_case)
def double_value(x):
    return x * 2

def calculate_sum(x, y):
    return x + y
# get_student_grade(score), validate_email(email), ...
```

### 3.8.2 Function ngắn gọn và tập trung

```python
# ❌ Function làm quá nhiều việc
def process_data(data):
    # Đọc dữ liệu
    # Làm sạch dữ liệu
    # Phân tích dữ liệu
    # Vẽ biểu đồ
    # Lưu kết quả
    # ... quá nhiều logic
    pass

# ✅ Chia nhỏ thành nhiều function
def read_data(file): ...
def clean_data(data): ...
def analyze_data(data): ...
def plot_results(results): ...
def save_results(results, file): ...
```

> `pass` và `...` là câu lệnh "rỗng", dùng làm chỗ giữ chỗ khi chưa viết thân hàm.

### 3.8.3 Sử dụng tham số mặc định hợp lý

```python
from scipy import stats

# ✅ Tham số mặc định có ý nghĩa
def calculate_statistics(data,
                         conf_level=0.95,   # Mức tin cậy 95%
                         digits=2):         # Làm tròn 2 chữ số
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]            # Bỏ qua NaN
    mean_val = data.mean()
    se = stats.sem(data)                    # Sai số chuẩn
    ci = stats.t.interval(conf_level, df=len(data) - 1, loc=mean_val, scale=se)

    result = {
        "mean": round(float(mean_val), digits),
        "ci_lower": round(float(ci[0]), digits),
        "ci_upper": round(float(ci[1]), digits),
    }
    return result

print(calculate_statistics([75, 82, 68, 91, 77, 85, 73]))
```

### 3.8.4 Validate input

```python
# ✅ Kiểm tra input trước khi xử lý
def safe_divide(a, b):
    # Kiểm tra kiểu dữ liệu
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Cả hai tham số phải là số")   # R: stop(...)

    # Kiểm tra chia cho 0
    if b == 0:
        raise ValueError("Không thể chia cho 0")

    return a / b

print(safe_divide(10, 2))   # 5.0
# safe_divide(10, 0)        # Lỗi ValueError: Không thể chia cho 0
# safe_divide("a", 2)       # Lỗi TypeError: Cả hai tham số phải là số
```

### 3.8.5 Viết documentation (docstring)

Trong Python, tài liệu của hàm được viết bằng **docstring** – chuỗi đặt ngay dưới dòng `def`. Xem tài liệu bằng `help(ten_ham)`.

```python
def calculate_bmi(weight, height):
    """Tính chỉ số BMI và phân loại.

    Parameters
    ----------
    weight : float
        Cân nặng (kg)
    height : float
        Chiều cao (m)

    Returns
    -------
    dict
        Dict chứa BMI và phân loại

    Examples
    --------
    >>> calculate_bmi(70, 1.75)
    {'bmi': 22.86, 'category': 'Bình thường'}
    """
    if weight <= 0 or height <= 0:
        raise ValueError("Cân nặng và chiều cao phải dương")

    bmi = weight / (height ** 2)

    if bmi < 18.5:
        category = "Thiếu cân"
    elif bmi < 25:
        category = "Bình thường"
    elif bmi < 30:
        category = "Thừa cân"
    else:
        category = "Béo phì"

    return {"bmi": round(bmi, 2), "category": category}

help(calculate_bmi)
```

---

## 3.9 Debugging Function

### 3.9.1 Sử dụng print() để debug

```python
def calculate_discount(price, discount_percent):
    print(f"Price: {price}")                       # Debug
    print(f"Discount: {discount_percent}")         # Debug

    discount_amount = price * (discount_percent / 100)
    print(f"Discount amount: {discount_amount}")   # Debug

    final_price = price - discount_amount
    return final_price

print(calculate_discount(1000, 10))
```

### 3.9.2 Sử dụng breakpoint()

```python
# Tạm dừng thực thi để kiểm tra (tương đương browser() trong R)
def debug_function(x, y):
    result = x + y
    breakpoint()   # Dừng tại đây để debug
    result = result * 2
    return result

# Khi chạy, trình gỡ lỗi pdb sẽ mở ra:
#   gõ tên biến để xem giá trị, n (next) để chạy dòng tiếp, c (continue) để chạy tiếp, q để thoát
# debug_function(5, 3)
```

> Trong VS Code / JupyterLab, cách tiện hơn là đặt **breakpoint** bằng cách click vào lề trái của dòng code rồi chọn *Debug Cell* / *Debug Python File*.

### 3.9.3 Sử dụng try/except xử lý lỗi

Tương đương `tryCatch()` trong R:

```python
def safe_log(x):
    try:
        result = math.log(x)
    except ValueError as e:        # Ví dụ: log của số âm
        result = f"Lỗi giá trị: {e}"
    except TypeError as e:         # Ví dụ: log của chuỗi
        result = f"Lỗi kiểu dữ liệu: {e}"
    return result

print(safe_log(10))     # 2.302585092994046
print(safe_log(-5))     # Lỗi giá trị: math domain error
print(safe_log("abc"))  # Lỗi kiểu dữ liệu: must be real number, not str
```

> **Khác biệt với R:** R trả về `NaN` kèm *cảnh báo* khi tính `log(-5)`. Hàm `math.log` của Python **báo lỗi** (`ValueError`), còn `np.log(-5)` trả về `nan` kèm `RuntimeWarning` – giống R hơn.

---

## BÀI TẬP THỰC HÀNH

### Bài tập 1: Function cơ bản
```python
# 1. Viết function tính diện tích hình chữ nhật
# Input: chiều dài, chiều rộng
# Output: diện tích

# 2. Viết function tính chu vi hình tròn
# Input: bán kính
# Output: chu vi (gợi ý: math.pi)

# 3. Viết function chuyển đổi nhiệt độ từ Celsius sang Fahrenheit
# Công thức: F = C * 9/5 + 32
```

### Bài tập 2: Function với validation
```python
# 1. Viết function kiểm tra số chẵn/lẻ
# Input: một số nguyên
# Output: "Chẵn" hoặc "Lẻ"
# Validate: input phải là số nguyên (gợi ý: isinstance(n, int))

# 2. Viết function tính điểm trung bình
# Input: list điểm số
# Output: điểm trung bình
# Validate: 
#   - Điểm phải từ 0 đến 10
#   - Loại bỏ giá trị None / NaN
```

### Bài tập 3: Function thống kê
```python
# 1. Viết function tính toán tổng quan
# Input: list/mảng số
# Output: dict(mean, median, sd, min, max, range)

# 2. Viết function tính chỉnh hợp A(n, r)
# Công thức: A(n,r) = n! / (n-r)!

# 3. Viết function tính tổ hợp C(n, r)
# Công thức: C(n,r) = n! / (r! * (n-r)!)
# Kiểm tra lại kết quả bằng math.comb(n, r)
```

### Bài tập 4: Function nâng cao
```python
# 1. Viết function tìm các số nguyên tố từ 1 đến n
# Input: n
# Output: list các số nguyên tố

# 2. Viết function tạo tam giác Pascal với n hàng
# Gợi ý: Sử dụng tổ hợp C(n, k)

# 3. Viết function phân loại sinh viên dựa vào điểm
# Input: điểm số
# Output: xếp loại (Xuất sắc, Giỏi, Khá, TB, Yếu)
# Kèm theo GPA scale 4.0
```

### Bài tập 5: Ứng dụng thực tế
```python
# 1. Viết function tính lương ròng
# Input: lương cơ bản, phụ cấp, số ngày làm việc, số giờ tăng ca
# Output: lương ròng sau thuế

# 2. Viết function chuẩn hóa điểm thi
# Input: mảng điểm thô
# Output: mảng điểm chuẩn hóa (0-100)
# Công thức: (điểm - min) / (max - min) * 100

# 3. Viết function phân tích dữ liệu sinh viên
# Input: DataFrame (tên, tuổi, điểm)
# Output: thống kê mô tả đầy đủ
```

---

## TÀI LIỆU THAM KHẢO

1. **Python Documentation - Defining Functions**: https://docs.python.org/3/tutorial/controlflow.html#defining-functions
2. **PEP 8 – Style Guide for Python Code**: https://peps.python.org/pep-0008/
3. **PEP 257 – Docstring Conventions**: https://peps.python.org/pep-0257/
4. **Real Python – Defining Your Own Python Function**: https://realpython.com/defining-your-own-python-function/

---

## TỔNG KẾT

### Những điểm cần nhớ:
1. ✅ Function giúp code dễ đọc, dễ bảo trì và tái sử dụng
2. ✅ Sử dụng tên function có ý nghĩa (động từ + danh từ, snake_case)
3. ✅ Tham số mặc định giúp function linh hoạt hơn
4. ✅ Luôn validate input trước khi xử lý (`raise ValueError(...)`)
5. ✅ Function nên ngắn gọn, tập trung vào một nhiệm vụ
6. ✅ Viết docstring cho function phức tạp
7. ✅ **Luôn dùng `return`** – Python không tự trả về biểu thức cuối như R

### Quy trình viết function tốt:
1. **Xác định mục đích**: Function này làm gì?
2. **Thiết kế input/output**: Cần tham số gì? Trả về gì?
3. **Viết code đơn giản**: Viết logic cơ bản trước
4. **Thêm validation**: Kiểm tra input hợp lệ
5. **Test kỹ lưỡng**: Thử với nhiều trường hợp khác nhau
6. **Refactor**: Cải thiện code, thêm tính năng
7. **Document**: Viết docstring hướng dẫn sử dụng

---

**Lưu ý quan trọng:**
- Function là nền tảng của lập trình Python
- Thực hành viết function thường xuyên để thành thạo
- Đọc code của người khác để học cách viết function tốt
- Các function trong bài học tiếp theo (Thống kê) đều sử dụng các nguyên tắc này

**Cập nhật**: Tháng 3/2026
