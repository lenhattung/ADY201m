# BÀI 10: ĐẠI SỐ TỔ HỢP VÀ XÁC SUẤT CƠ BẢN

## Mục tiêu học tập
- Hiểu và tính được giai thừa, hoán vị, chỉnh hợp, tổ hợp
- Phân biệt được hoán vị, chỉnh hợp và tổ hợp (có lặp và không lặp)
- Nắm vững các quy tắc đếm cơ bản
- Áp dụng đại số tổ hợp vào bài toán xác suất thực tế
- Sử dụng Python để tính toán các bài toán tổ hợp
- Giải quyết được các bài toán đếm trong thực tế

> **Công cụ Python dùng trong bài:** module `math` có sẵn (`math.factorial`, `math.perm`, `math.comb` – yêu cầu Python ≥ 3.8), module `itertools` để **liệt kê** các cách sắp xếp/chọn, và `scipy.special` khi cần tính trên cả mảng.

---

## 10.1 Giai thừa (Factorial)

### 10.1.1 Giai thừa là gì?

**Giai thừa** của một số nguyên dương n, ký hiệu là **n!**, là tích của tất cả các số nguyên dương từ 1 đến n.

**Công thức:**
```
n! = 1 × 2 × 3 × ... × n
```

**Quy ước đặc biệt:**
```
0! = 1  (theo quy ước)
```

### 10.1.2 Ví dụ tính giai thừa

**Ví dụ 1:** Tính 5!
```
5! = 1 × 2 × 3 × 4 × 5 = 120
```

**Ví dụ 2:** Tính 3!
```
3! = 1 × 2 × 3 = 6
```

**Ví dụ 3:** Tính 0!
```
0! = 1  (theo quy ước)
```

### 10.1.3 Tính giai thừa trong Python

```python
import math

# Tính 5!
print(math.factorial(5))   # Kết quả: 120

# Tính 0!
print(math.factorial(0))   # Kết quả: 1

# Tính 10!
print(f"{math.factorial(10):,}")   # Kết quả: 3,628,800

# Tính giai thừa cho nhiều số cùng lúc
print([math.factorial(n) for n in range(1, 11)])

# Hoặc dùng scipy cho cả mảng (R: factorial(1:10))
from scipy.special import factorial
import numpy as np
print(factorial(np.arange(1, 11), exact=True))
```

> **Ưu điểm của Python:** Số nguyên trong Python không giới hạn độ lớn, nên `math.factorial(50)` cho kết quả **chính xác tuyệt đối** (R trả về số thực xấp xỉ `3.041409e+64`).

---

## 10.2 Quy tắc đếm cơ bản

### 10.2.1 Quy tắc cộng (Addition Rule)

**Quy tắc:** Nếu có **m** cách thực hiện việc A và **n** cách thực hiện việc B, và hai việc **không thể xảy ra đồng thời**, thì có **m + n** cách thực hiện A hoặc B.

**Ví dụ:** 
- Trong lớp có 15 nam và 12 nữ
- Chọn 1 học sinh làm lớp trưởng
- Số cách chọn = 15 + 12 = 27 cách

```python
# Quy tắc cộng
male = 15
female = 12
total_ways = male + female
print("Số cách chọn:", total_ways)  # 27
```

### 10.2.2 Quy tắc nhân (Multiplication Rule)

**Quy tắc:** Nếu có **m** cách thực hiện việc A và **n** cách thực hiện việc B, và hai việc **độc lập**, thì có **m × n** cách thực hiện A và B.

**Ví dụ:** 
- Có 3 áo và 4 quần
- Số cách phối đồ = 3 × 4 = 12 cách

```python
# Quy tắc nhân
shirts = 3
pants = 4
outfits = shirts * pants
print("Số cách phối đồ:", outfits)  # 12

# Kiểm chứng bằng cách liệt kê tất cả (itertools.product = tích Descartes)
from itertools import product
ao = ["Áo1", "Áo2", "Áo3"]
quan = ["Q1", "Q2", "Q3", "Q4"]
cac_bo = list(product(ao, quan))
print(len(cac_bo), cac_bo[:3])
```

### 10.2.3 Ví dụ tổng hợp

**Ví dụ:** Mật khẩu gồm 1 chữ cái (A-Z) và 3 chữ số (0-9). Hỏi có bao nhiêu mật khẩu?

**Giải:**
- Chữ cái: 26 cách
- Chữ số thứ 1: 10 cách
- Chữ số thứ 2: 10 cách
- Chữ số thứ 3: 10 cách
- Tổng: 26 × 10 × 10 × 10 = 26,000

```python
passwords = 26 * 10 * 10 * 10
print("Số mật khẩu:", passwords)  # 26000
```

---

## 10.3 Hoán vị (Permutation)

### 10.3.1 Hoán vị không lặp

**Hoán vị** là cách sắp xếp **tất cả** các phần tử theo thứ tự, các phần tử **khác nhau**.

**Ký hiệu:** P(n)

**Công thức:**
```
P(n) = n!
```

**Ví dụ:** Có bao nhiêu cách sắp xếp 3 người A, B, C?
```
P(3) = 3! = 6 cách
ABC, ACB, BAC, BCA, CAB, CBA
```

```python
from itertools import permutations

# Hoán vị 3 người
print(math.factorial(3))  # 6

# Liệt kê tất cả các hoán vị
for p in permutations(["A", "B", "C"]):
    print("".join(p), end=" ")
```

### 10.3.2 Hoán vị lặp (Permutation with Repetition)

**Hoán vị lặp** là hoán vị khi có các phần tử **giống nhau**.

**Công thức:**
```
P(n; n₁, n₂, ..., nₖ) = n! / (n₁! × n₂! × ... × nₖ!)
```

Trong đó:
- n: tổng số phần tử
- n₁, n₂, ..., nₖ: số lần lặp của mỗi phần tử

**Ví dụ 1:** Có bao nhiêu cách sắp xếp các chữ cái trong từ "BANANA"?

**Phân tích:**
- Tổng: 6 chữ cái
- B: 1 lần
- A: 3 lần
- N: 2 lần

```
P = 6! / (1! × 3! × 2!) = 720 / (1 × 6 × 2) = 60
```

```python
# Hoán vị lặp cho BANANA
total_letters = 6
b_count = 1
a_count = 3
n_count = 2

ways = math.factorial(total_letters) // (math.factorial(b_count) *
                                         math.factorial(a_count) *
                                         math.factorial(n_count))
print("Số cách sắp xếp BANANA:", ways)  # 60

# Cách tổng quát: tự đếm số lần lặp bằng Counter
from collections import Counter
def hoan_vi_lap(word):
    counts = Counter(word)                 # {'A': 3, 'N': 2, 'B': 1}
    result = math.factorial(len(word))
    for c in counts.values():
        result //= math.factorial(c)
    return result

print(hoan_vi_lap("BANANA"))                # 60
print(len(set(permutations("BANANA"))))     # 60 – kiểm chứng bằng liệt kê
```

> Dùng phép chia nguyên `//` để kết quả là số nguyên (phép chia `/` cho kết quả số thực `60.0`).

**Ví dụ 2:** Có bao nhiêu cách sắp xếp 10 viên bi: 5 đỏ, 3 xanh, 2 vàng?

```python
total_balls = 10
red = 5
blue = 3
yellow = 2

ways = math.factorial(total_balls) // (math.factorial(red) *
                                       math.factorial(blue) *
                                       math.factorial(yellow))
print("Số cách sắp xếp:", ways)  # 2520
```

---

## 10.4 Chỉnh hợp (Arrangement)

### 10.4.1 Chỉnh hợp không lặp

**Chỉnh hợp** là cách chọn và sắp xếp **r** phần tử từ **n** phần tử khác nhau, **có quan tâm thứ tự**.

**Ký hiệu:** A(n, r)

**Công thức:**
```
A(n, r) = n! / (n - r)!
```

**Ví dụ:** Chọn 3 người từ 5 người để làm Lớp trưởng, Lớp phó, Thư ký.

```
A(5, 3) = 5! / 2! = 60
```

```python
# Hàm tự viết tính chỉnh hợp
def chinh_hop(n, r):
    return math.factorial(n) // math.factorial(n - r)

print(chinh_hop(5, 3))    # 60

# Python có sẵn hàm math.perm(n, r)
print(math.perm(5, 3))    # 60
```

### 10.4.2 Chỉnh hợp lặp (Arrangement with Repetition)

**Chỉnh hợp lặp** là chọn r phần tử từ n phần tử, mỗi phần tử **có thể chọn nhiều lần**.

**Công thức:**
```
A'(n, r) = nʳ
```

**Ví dụ 1:** Mật khẩu gồm 4 chữ số (0-9), mỗi chữ số có thể lặp lại. Có bao nhiêu mật khẩu?

```
A'(10, 4) = 10⁴ = 10,000
```

```python
# Chỉnh hợp lặp
n = 10  # Chữ số 0-9
r = 4   # 4 vị trí
ways = n ** r
print("Số mật khẩu:", ways)  # 10000
```

**Ví dụ 2:** Có bao nhiêu số có 3 chữ số (từ 0-9, có thể lặp)?

```python
# Từ 000 đến 999
ways = 10 ** 3
print("Số có 3 chữ số (kể cả 000):", ways)  # 1000
```

**So sánh:**

```python
# Không lặp (các chữ số khác nhau)
print(math.perm(10, 3))  # 720

# Có lặp (các chữ số có thể giống nhau)
print(10 ** 3)           # 1000
```

---

## 10.5 Tổ hợp (Combination)

### 10.5.1 Tổ hợp không lặp

**Tổ hợp** là cách chọn **r** phần tử từ **n** phần tử, **không quan tâm thứ tự**.

**Ký hiệu:** C(n, r)

**Công thức:**
```
C(n, r) = n! / (r! × (n - r)!)
```

**Ví dụ:** Chọn 3 học sinh từ 5 học sinh vào đội thi.

```
C(5, 3) = 10
```

```python
from itertools import combinations

print(math.comb(5, 3))  # 10  (R: choose(5, 3))

# Liệt kê các cách chọn
hs = ["An", "Bình", "Chi", "Dũng", "Em"]
for nhom in combinations(hs, 3):
    print(nhom)
```

### 10.5.2 Tổ hợp lặp (Combination with Repetition)

**Tổ hợp lặp** là chọn r phần tử từ n phần tử, mỗi phần tử **có thể chọn nhiều lần**, không quan tâm thứ tự.

**Công thức:**
```
C'(n, r) = C(n + r - 1, r) = (n + r - 1)! / (r! × (n - 1)!)
```

**Ví dụ 1:** Mua 5 quả táo từ 3 loại (táo đỏ, táo xanh, táo vàng). Có bao nhiêu cách mua?

**Phân tích:**
- n = 3 (3 loại táo)
- r = 5 (mua 5 quả)
- Có thể mua cùng loại

```
C'(3, 5) = C(3 + 5 - 1, 5) = C(7, 5) = 21
```

```python
# Tổ hợp lặp
def to_hop_lap(n, r):
    return math.comb(n + r - 1, r)

n = 3  # 3 loại táo
r = 5  # 5 quả
ways = to_hop_lap(n, r)
print("Số cách mua:", ways)  # 21
```

**Ví dụ 2:** Có bao nhiêu cách chọn 4 viên kẹo từ 3 loại kẹo (có thể chọn cùng loại)?

```python
from itertools import combinations_with_replacement

n = 3  # 3 loại kẹo
r = 4  # 4 viên
ways = to_hop_lap(n, r)
print("Số cách chọn:", ways)  # 15

# Kiểm chứng bằng cách liệt kê
cach_chon = list(combinations_with_replacement("ABC", 4))
print(len(cach_chon))
print(["".join(c) for c in cach_chon])
```

**Giải thích bằng ví dụ cụ thể:**
- Loại A, B, C
- Chọn 4 viên, ví dụ: AAAB, AABC, BBBC, ...
- C'(3, 4) = C(6, 4) = 15 cách

---

## 10.6 Bảng tổng hợp

### 10.6.1 So sánh tất cả các loại

| Loại | Ký hiệu | Công thức | Thứ tự | Lặp | Python | Liệt kê (`itertools`) | Ví dụ |
|------|---------|-----------|---------|-----|--------|--------|-------|
| **Hoán vị** | P(n) | n! | Có | Không | `math.factorial(n)` | `permutations(x)` | Xếp 5 người vào 5 ghế |
| **Hoán vị lặp** | P(n;n₁,n₂) | n!/(n₁!×n₂!...) | Có | Có | tự viết | `set(permutations(x))` | Xếp BANANA |
| **Chỉnh hợp** | A(n,r) | n!/(n-r)! | Có | Không | `math.perm(n, r)` | `permutations(x, r)` | Chọn 3 từ 5 vào 3 vị trí |
| **Chỉnh hợp lặp** | A'(n,r) | nʳ | Có | Có | `n ** r` | `product(x, repeat=r)` | Mật khẩu 4 số |
| **Tổ hợp** | C(n,r) | n!/(r!(n-r)!) | Không | Không | `math.comb(n, r)` | `combinations(x, r)` | Chọn 3 từ 5 nhóm |
| **Tổ hợp lặp** | C'(n,r) | C(n+r-1,r) | Không | Có | `math.comb(n+r-1, r)` | `combinations_with_replacement(x, r)` | Mua 5 táo từ 3 loại |

### 10.6.2 Cách nhận biết nhanh

**Bước 1: Có quan tâm THỨ TỰ không?**
- **Có** → Hoán vị hoặc Chỉnh hợp
- **Không** → Tổ hợp

**Bước 2: Chọn TẤT CẢ hay MỘT PHẦN?**
- **Tất cả** → Hoán vị
- **Một phần** → Chỉnh hợp hoặc Tổ hợp

**Bước 3: Có LẶP không?**
- **Có lặp** → Thêm dấu ' hoặc công thức lặp
- **Không lặp** → Công thức thường

### 10.6.3 Sơ đồ quyết định

```
                    Bài toán đếm
                         |
        Có quan tâm THỨ TỰ không?
       /                            \
     CÓ                           KHÔNG
      |                              |
Chọn TẤT CẢ hay PHẦN?         Có LẶP không?
   /        \                   /         \
TẤT CẢ    PHẦN               KHÔNG       CÓ
  |         |                  |           |
HOÁN VỊ  Có LẶP?            TỔ HỢP    TỔ HỢP LẶP
        /     \              C(n,r)    C'(n,r)
     KHÔNG    CÓ
       |       |
   CHỈNH HỢP  CHỈNH HỢP LẶP
   A(n,r)     nʳ
```

---

## 10.7 Ví dụ tổng hợp

### Ví dụ 1: Mật khẩu

**Câu hỏi:** Mật khẩu gồm 6 ký tự từ a-z (26 chữ cái). Tính số mật khẩu nếu:
a) Các ký tự có thể lặp
b) Các ký tự không được lặp

**Giải:**

```python
# a) Có lặp - Chỉnh hợp lặp
passwords_with_rep = 26 ** 6
print(f"a) Có lặp: {passwords_with_rep:,}")

# b) Không lặp - Chỉnh hợp
passwords_no_rep = math.perm(26, 6)
print(f"b) Không lặp: {passwords_no_rep:,}")
```

### Ví dụ 2: Xếp bi

**Câu hỏi:** Có 8 viên bi: 3 đỏ, 3 xanh, 2 vàng. Có bao nhiêu cách xếp thành hàng?

**Giải:** Hoán vị lặp

```python
total = 8
red = 3
blue = 3
yellow = 2

ways = math.factorial(total) // (math.factorial(red) *
                                 math.factorial(blue) *
                                 math.factorial(yellow))
print("Số cách xếp:", ways)  # 560
```

### Ví dụ 3: Mua hoa quả

**Câu hỏi:** Mua 6 quả từ 4 loại (cam, táo, chuối, nho). Có bao nhiêu cách mua?

**Giải:** Tổ hợp lặp

```python
n = 4  # 4 loại
r = 6  # 6 quả
ways = to_hop_lap(n, r)
print("Số cách mua:", ways)  # 84
```

### Ví dụ 4: Xếp sách

**Câu hỏi:** 
a) Xếp 5 quyển sách khác nhau
b) Xếp 5 quyển sách trong đó có 2 quyển giống nhau

**Giải:**

```python
# a) Hoán vị
ways_a = math.factorial(5)
print("a) Sách khác nhau:", ways_a)  # 120

# b) Hoán vị lặp
ways_b = math.factorial(5) // math.factorial(2)
print("b) Có 2 sách giống:", ways_b)  # 60
```

---

## 10.8 Ứng dụng trong Xác suất

### 10.8.1 Xác suất cơ bản

**Công thức:**
```
P(A) = Số kết quả thuận lợi / Tổng số kết quả có thể
```

### 10.8.2 Ví dụ về xác suất

**Ví dụ 1:** Xác suất rút 4 quân Át từ bộ bài 52 lá

```python
# Tổng số cách chọn 4 lá
total_ways = math.comb(52, 4)

# Số cách chọn 4 quân Át
ace_ways = math.comb(4, 4)

# Xác suất
prob = ace_ways / total_ways
print("Xác suất rút 4 Át:", prob)                # dạng khoa học: 3.69...e-06
print(f"Xác suất: {prob:.10f}")                   # dạng thập phân

# Dùng Fraction để có kết quả phân số chính xác
from fractions import Fraction
print("Phân số:", Fraction(ace_ways, total_ways))  # 1/270725
```

**Ví dụ 2:** Xác suất có đúng 2 quân Át trong 5 lá bài

```python
# Chọn 2 Át từ 4 Át
ways_2_aces = math.comb(4, 2)

# Chọn 3 lá khác từ 48 lá
ways_3_others = math.comb(48, 3)

# Tổng số cách chọn 5 lá
total_ways = math.comb(52, 5)

# Xác suất
prob = (ways_2_aces * ways_3_others) / total_ways
print("Xác suất có đúng 2 Át:", round(prob, 4))
```

**Ví dụ 3:** Xổ số - Chọn 6 số từ 45 số

```python
# Tổng số cách chọn
total = math.comb(45, 6)
print(f"Tổng số tổ hợp: {total:,}")

# Xác suất trúng jackpot
prob_jackpot = 1 / total
print("Xác suất trúng:", prob_jackpot)
print(f"Tỷ lệ: 1 trên {total:,}")
```

**Kiểm chứng bằng mô phỏng (Monte Carlo):** Ta có thể "rút bài" ngẫu nhiên rất nhiều lần và đếm tỷ lệ thành công.

```python
import numpy as np

rng = np.random.default_rng(42)
bo_bai = np.array([1] * 4 + [0] * 48)     # 1 = quân Át, 0 = quân khác
n_sim = 100_000

so_at = np.array([rng.choice(bo_bai, 5, replace=False).sum() for _ in range(n_sim)])
print("Mô phỏng  P(đúng 2 Át):", np.mean(so_at == 2))
print("Lý thuyết P(đúng 2 Át):", round(math.comb(4, 2) * math.comb(48, 3) / math.comb(52, 5), 4))
```

---

## 10.9 Tam giác Pascal

### 10.9.1 Tam giác Pascal

**Tam giác Pascal** hiển thị các giá trị C(n, r).

```python
# Vẽ tam giác Pascal
def pascal_triangle(n):
    for i in range(n + 1):
        row = [math.comb(i, j) for j in range(i + 1)]
        print(" " * (n - i) * 2 + "   ".join(str(v) for v in row))

pascal_triangle(6)
```

### 10.9.2 Tính chất

**Tính chất 1:** Tổng hàng n = 2^n
```python
n = 5
row_sum = sum(math.comb(n, k) for k in range(n + 1))
print(f"Tổng hàng {n}: {row_sum}")
print(f"2^{n} = {2 ** n}")
```

**Tính chất 2:** Đối xứng C(n, r) = C(n, n-r)
```python
print("C(10, 3) =", math.comb(10, 3))
print("C(10, 7) =", math.comb(10, 7))
```

---

## BÀI TẬP THỰC HÀNH

### Bài tập 1: Phân loại

Xác định mỗi bài toán sau dùng công thức nào:

1. Sắp xếp 7 quyển sách khác nhau
2. Sắp xếp các chữ cái trong từ "MATHEMATICS"
3. Mật khẩu 5 chữ số, mỗi chữ số từ 0-9, có thể lặp
4. Chọn 4 học sinh từ 20 học sinh vào đội thi
5. Chọn 3 viên kẹo từ 5 loại, có thể chọn cùng loại

**Đáp án:**
```python
# 1. Hoán vị
print(math.factorial(7))

# 2. Hoán vị lặp (M:2, A:2, T:2, H:1, E:1, I:1, C:1, S:1)
print(math.factorial(11) // (math.factorial(2) * math.factorial(2) * math.factorial(2)))
print(hoan_vi_lap("MATHEMATICS"))   # Kiểm tra lại bằng hàm tự viết

# 3. Chỉnh hợp lặp
print(10 ** 5)

# 4. Tổ hợp
print(math.comb(20, 4))

# 5. Tổ hợp lặp
print(to_hop_lap(5, 3))
```

### Bài tập 2: Mật khẩu

Mật khẩu gồm 8 ký tự. Tính số mật khẩu nếu:
1. Chỉ dùng chữ số 0-9, có thể lặp
2. Chỉ dùng chữ số 0-9, không lặp
3. Dùng cả chữ và số (a-z, 0-9), có thể lặp

### Bài tập 3: Xếp người

1. Có bao nhiêu cách xếp 5 người A, B, C, D, E thành hàng ngang?
2. Có bao nhiêu cách nếu A và B phải đứng cạnh nhau?
3. Có bao nhiêu cách nếu A và B không được đứng cạnh nhau?

*Gợi ý:* Kiểm chứng đáp án bằng cách dùng `itertools.permutations("ABCDE")` rồi đếm các hoán vị thỏa điều kiện.

### Bài tập 4: Phân phối

Có 10 viên bi giống nhau, chia cho 3 bạn. Mỗi bạn nhận ít nhất 1 viên. Có bao nhiêu cách chia?

### Bài tập 5: Tổ hợp lặp

1. Có bao nhiêu cách mua 8 quả từ 4 loại trái cây?
2. Phương trình x₁ + x₂ + x₃ = 10, với x₁, x₂, x₃ ≥ 0, có bao nhiêu nghiệm nguyên không âm?

---

## CÂU HỎI ÔN TẬP

1. Phân biệt Hoán vị, Chỉnh hợp và Tổ hợp?
2. Khi nào dùng công thức có lặp?
3. Giải thích sự khác biệt giữa A(n,r) và nʳ?
4. Tại sao C'(n,r) = C(n+r-1, r)?
5. Quy tắc cộng và quy tắc nhân khác nhau như thế nào?
6. Cho ví dụ thực tế về mỗi loại công thức?

---

## TÀI LIỆU THAM KHẢO

1. **Python Documentation**: [`math` module](https://docs.python.org/3/library/math.html) (`factorial`, `perm`, `comb`), [`itertools` module](https://docs.python.org/3/library/itertools.html)
2. **Toán học rời rạc**: Giáo trình Đại học
3. **Xác suất thống kê**: Giáo trình cơ bản

---

## TỔNG KẾT

### Công thức tổng hợp:

| Loại | Công thức | Khi nào dùng | Python |
|------|-----------|--------------|--------|
| **Giai thừa** | n! | Sắp xếp tất cả | `math.factorial(n)` |
| **Hoán vị** | P(n) = n! | Xếp tất cả, không lặp | `math.factorial(n)` |
| **Hoán vị lặp** | n!/(n₁!×n₂!...) | Xếp tất cả, có phần tử giống nhau | tự viết (`Counter`) |
| **Chỉnh hợp** | A(n,r) = n!/(n-r)! | Chọn r từ n, có thứ tự, không lặp | `math.perm(n, r)` |
| **Chỉnh hợp lặp** | nʳ | Chọn r từ n, có thứ tự, có lặp | `n ** r` |
| **Tổ hợp** | C(n,r) = n!/(r!(n-r)!) | Chọn r từ n, không thứ tự, không lặp | `math.comb(n, r)` |
| **Tổ hợp lặp** | C(n+r-1,r) | Chọn r từ n, không thứ tự, có lặp | `math.comb(n+r-1, r)` |

### Lưu ý quan trọng:

- **Thứ tự quan trọng** → Hoán vị/Chỉnh hợp
- **Thứ tự không quan trọng** → Tổ hợp
- **Có lặp** → Dùng công thức lặp
- **Không lặp** → Dùng công thức thường

---

**Cập nhật**: Tháng 3/2026
