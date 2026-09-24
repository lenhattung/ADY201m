# BÀI 11: THỐNG KÊ MÔ TẢ

## Mục tiêu học tập
- Hiểu khái niệm và vai trò của thống kê mô tả
- Tính và giải thích các số đo xu hướng trung tâm (mean, median, mode)
- Tính và giải thích các số đo độ phân tán (variance, SD, range, IQR)
- Tính và giải thích các số đo hình dạng (skewness, kurtosis)
- Tạo và phân tích bảng tần số
- Sử dụng Python (NumPy, pandas, SciPy) để tính toán và trực quan hóa thống kê mô tả
- Áp dụng thống kê mô tả vào bài toán thực tế

> **Chuẩn bị:**
> ```python
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> from scipy import stats
> ```

---

## 11.1 Thống kê mô tả là gì?

### 11.1.1 Định nghĩa

**Thống kê mô tả** (Descriptive Statistics) là tập hợp các phương pháp để **tóm tắt, tổ chức và trình bày dữ liệu** một cách có ý nghĩa.

**Mục đích:**
- **Tóm tắt** dữ liệu thành các con số dễ hiểu
- **Mô tả** đặc điểm chính của tập dữ liệu
- **Trực quan hóa** dữ liệu bằng biểu đồ
- **So sánh** các nhóm dữ liệu khác nhau

### 11.1.2 Ví dụ minh họa

**Tình huống:** Điểm thi của 10 sinh viên
```
85, 90, 78, 92, 88, 95, 80, 89, 91, 87
```

**Thay vì nhìn cả 10 số, ta có thể tóm tắt:**
- Điểm trung bình: 87.5
- Điểm cao nhất: 95
- Điểm thấp nhất: 78
- Hầu hết sinh viên đạt từ 80-92 điểm

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Ví dụ trong Python
scores = np.array([85, 90, 78, 92, 88, 95, 80, 89, 91, 87])

# Các thống kê cơ bản
print("Điểm trung bình:", scores.mean())
print("Điểm cao nhất:", scores.max())
print("Điểm thấp nhất:", scores.min())
print("Độ lệch chuẩn:", round(scores.std(ddof=1), 2))   # ddof=1: độ lệch chuẩn MẪU
```

> ⚠️ **Quan trọng – tham số `ddof`:** Hàm `sd()` và `var()` của R chia cho **(n − 1)** (thống kê mẫu). Trong Python:
> * NumPy (`np.std`, `np.var`, `arr.std()`) mặc định chia cho **n** → phải thêm `ddof=1`.
> * pandas (`Series.std()`, `Series.var()`) mặc định đã chia cho **(n − 1)** – giống R.

### 11.1.3 Các loại thống kê mô tả

Thống kê mô tả gồm 3 nhóm chính:

1. **Số đo xu hướng trung tâm** (Central Tendency)
   - Trung bình (Mean)
   - Trung vị (Median)
   - Mode (Yếu vị)

2. **Số đo độ phân tán** (Dispersion/Spread)
   - Phương sai (Variance)
   - Độ lệch chuẩn (Standard Deviation)
   - Khoảng biến thiên (Range)
   - IQR (Interquartile Range)

3. **Số đo hình dạng phân phối** (Shape)
   - Độ lệch (Skewness)
   - Độ nhọn (Kurtosis)

---

## 11.2 Số đo xu hướng trung tâm

### 11.2.1 Trung bình (Mean)

**Định nghĩa:** Trung bình là tổng các giá trị chia cho số lượng giá trị.

**Công thức:**
```
x̄ = (x₁ + x₂ + ... + xₙ) / n = Σxᵢ / n
```

**Ví dụ 1:** Tính điểm trung bình
```
Điểm: 8, 7, 9, 6, 8
x̄ = (8 + 7 + 9 + 6 + 8) / 5 = 38 / 5 = 7.6
```

```python
# Tính trung bình trong Python
scores = np.array([8, 7, 9, 6, 8])
mean_score = np.mean(scores)
print("Điểm trung bình:", mean_score)  # 7.6
```

**Ví dụ 2:** Thu nhập trung bình
```
Thu nhập (triệu): 10, 12, 15, 11, 13, 50
x̄ = (10 + 12 + 15 + 11 + 13 + 50) / 6 = 18.5
```

```python
income = np.array([10, 12, 15, 11, 13, 50])
mean_income = np.mean(income)
print("Thu nhập TB:", mean_income, "triệu")  # 18.5
```

**Lưu ý:** Trung bình **nhạy cảm với giá trị ngoại lệ** (outliers). Trong ví dụ trên, giá trị 50 làm trung bình cao hơn rất nhiều so với phần lớn dữ liệu.

### 11.2.2 Trung vị (Median)

**Định nghĩa:** Trung vị là giá trị nằm chính giữa khi dữ liệu được sắp xếp theo thứ tự.

**Cách tính:**
1. Sắp xếp dữ liệu theo thứ tự tăng dần
2. Nếu n lẻ: Median = giá trị ở vị trí (n+1)/2
3. Nếu n chẵn: Median = trung bình của 2 giá trị giữa

**Ví dụ 1:** n lẻ
```
Dữ liệu: 8, 7, 9, 6, 8
Sắp xếp: 6, 7, 8, 8, 9
Median = 8 (vị trí thứ 3)
```

```python
data = np.array([8, 7, 9, 6, 8])
median_value = np.median(data)
print("Trung vị:", median_value)  # 8.0
```

**Ví dụ 2:** n chẵn
```
Dữ liệu: 10, 12, 15, 11, 13, 50
Sắp xếp: 10, 11, 12, 13, 15, 50
Median = (12 + 13) / 2 = 12.5
```

```python
income = np.array([10, 12, 15, 11, 13, 50])
median_income = np.median(income)
print("Trung vị:", median_income, "triệu")  # 12.5
```

**So sánh Mean vs Median:**

```python
income = np.array([10, 12, 15, 11, 13, 50])
print("Mean:", np.mean(income))      # 18.5
print("Median:", np.median(income))  # 12.5

# Median không bị ảnh hưởng bởi outlier (50)
# Mean bị kéo lên bởi outlier
```

**Khi nào dùng Median?**
- Khi dữ liệu có **outliers** (giá trị cực đoan)
- Thu nhập, giá nhà (thường có outliers)
- Dữ liệu **lệch** (không đối xứng)

### 11.2.3 Mode (Yếu vị)

**Định nghĩa:** Mode là giá trị xuất hiện **nhiều nhất** trong tập dữ liệu.

**Ví dụ 1:** Mode duy nhất
```
Điểm: 8, 7, 9, 8, 6, 8, 7
Mode = 8 (xuất hiện 3 lần)
```

```python
# R không có hàm mode sẵn, nhưng Python/pandas thì có
scores = pd.Series([8, 7, 9, 8, 6, 8, 7])
print("Mode:", scores.mode().tolist())   # [8]

# Tự viết hàm (trả về MỘT mode – giá trị xuất hiện nhiều nhất đầu tiên)
def get_mode(x):
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts)]

print("Mode:", get_mode([8, 7, 9, 8, 6, 8, 7]))   # 8
```

**Ví dụ 2:** Nhiều mode (Multimodal)
```
Điểm: 8, 7, 9, 8, 7, 6
Mode = 8 và 7 (cả hai xuất hiện 2 lần)
```

```python
import statistics
print(pd.Series([8, 7, 9, 8, 7, 6]).mode().tolist())   # [7, 8]
print(statistics.multimode([8, 7, 9, 8, 7, 6]))        # [8, 7]
```

**Ví dụ 3:** Không có mode
```
Điểm: 8, 7, 9, 6, 5
Không có mode (tất cả xuất hiện 1 lần)
```

> Lưu ý: Khi mọi giá trị chỉ xuất hiện 1 lần, `pd.Series.mode()` trả về **toàn bộ** các giá trị – cần tự kiểm tra trường hợp này.

**Khi nào dùng Mode?**
- Dữ liệu **phân loại** (categorical): màu sắc, kích cỡ áo
- Tìm giá trị **phổ biến nhất**

```python
# Ví dụ: Kích cỡ áo bán chạy nhất
sizes = pd.Series(["M", "L", "M", "S", "M", "L", "M", "XL"])
print(sizes.value_counts())
# M xuất hiện nhiều nhất → Mode = M
print("Mode:", sizes.mode()[0])
```

### 11.2.4 So sánh Mean, Median, Mode

```python
# Tạo dữ liệu mẫu
data = np.array([5, 6, 6, 7, 7, 7, 8, 8, 9, 25])  # 25 là outlier

print("=== SO SÁNH ===")
print("Mean:", np.mean(data))      # 8.8
print("Median:", np.median(data))  # 7.0
print("Mode:", get_mode(data))     # 7

# Nhận xét:
# - Mean cao nhất (bị ảnh hưởng bởi outlier 25)
# - Median và Mode gần nhau hơn (đại diện tốt hơn)
```

**Bảng tóm tắt:**

| Chỉ số | Ưu điểm | Nhược điểm | Khi nào dùng |
|--------|---------|------------|--------------|
| **Mean** | Sử dụng tất cả dữ liệu | Nhạy cảm với outliers | Dữ liệu đối xứng, không có outliers |
| **Median** | Không bị ảnh hưởng outliers | Không dùng hết dữ liệu | Dữ liệu có outliers, dữ liệu lệch |
| **Mode** | Dễ hiểu, dùng cho categorical | Có thể không tồn tại hoặc nhiều mode | Dữ liệu phân loại |

---

## 11.3 Số đo độ phân tán

### 11.3.1 Tại sao cần đo độ phân tán?

Hai tập dữ liệu có thể có **cùng trung bình** nhưng **độ phân tán khác nhau**.

**Ví dụ:**
```
Lớp A: 70, 75, 80, 85, 90  → Mean = 80
Lớp B: 50, 60, 80, 100, 110 → Mean = 80
```

Cả hai lớp đều có điểm TB = 80, nhưng:
- Lớp A: Điểm **tập trung** quanh 80
- Lớp B: Điểm **phân tán** rộng

```python
class_A = np.array([70, 75, 80, 85, 90])
class_B = np.array([50, 60, 80, 100, 110])

print("Mean A:", class_A.mean())  # 80.0
print("Mean B:", class_B.mean())  # 80.0

print("SD A:", round(class_A.std(ddof=1), 1))  # 7.9
print("SD B:", round(class_B.std(ddof=1), 1))  # 25.5
```

### 11.3.2 Khoảng biến thiên (Range)

**Định nghĩa:** Range là khoảng cách giữa giá trị **lớn nhất** và **nhỏ nhất**.

**Công thức:**
```
Range = Max - Min
```

**Ví dụ:**
```
Điểm: 78, 82, 90, 85, 88
Range = 90 - 78 = 12
```

```python
scores = np.array([78, 82, 90, 85, 88])

# Cách 1
range_value = scores.max() - scores.min()
print("Range:", range_value)  # 12

# Cách 2: np.ptp (peak to peak)
print("Min:", scores.min(), ", Max:", scores.max())
print("Range:", np.ptp(scores))
```

**Ưu điểm:** Dễ tính, dễ hiểu

**Nhược điểm:** Chỉ dùng 2 giá trị (min, max), nhạy cảm với outliers

### 11.3.3 Phương sai (Variance)

**Định nghĩa:** Phương sai đo **mức độ phân tán** của dữ liệu quanh trung bình.

**Công thức (mẫu):**
```
s² = Σ(xᵢ - x̄)² / (n - 1)
```

**Giải thích:**
- (xᵢ - x̄): Độ lệch của mỗi giá trị so với trung bình
- (xᵢ - x̄)²: Bình phương để tránh âm
- Chia cho (n-1): Để có ước lượng **không chệch** cho phương sai tổng thể

**Ví dụ tính tay:**
```
Dữ liệu: 4, 6, 8
Mean = (4 + 6 + 8) / 3 = 6

(4 - 6)² = 4
(6 - 6)² = 0
(8 - 6)² = 4

s² = (4 + 0 + 4) / (3 - 1) = 8 / 2 = 4
```

```python
data = np.array([4, 6, 8])
variance = np.var(data, ddof=1)
print("Phương sai:", variance)  # 4.0

# Tính tay
mean_val = data.mean()
squared_diff = (data - mean_val) ** 2
variance_manual = squared_diff.sum() / (len(data) - 1)
print("Phương sai (tay):", variance_manual)  # 4.0

# Nếu quên ddof=1:
print("np.var mặc định (chia n):", np.var(data))   # 2.67 – KHÁC với R!
```

**Ý nghĩa:**
- Phương sai **nhỏ**: Dữ liệu tập trung gần trung bình
- Phương sai **lớn**: Dữ liệu phân tán rộng

### 11.3.4 Độ lệch chuẩn (Standard Deviation)

**Định nghĩa:** Độ lệch chuẩn là **căn bậc hai** của phương sai.

**Công thức:**
```
s = √s²
```

**Tại sao dùng SD thay vì Variance?**
- SD có **cùng đơn vị** với dữ liệu gốc
- Dễ **giải thích** hơn

**Ví dụ:**
```
Điểm thi (0-100):
- Variance = 225
- SD = √225 = 15

Giải thích: Điểm sinh viên lệch trung bình khoảng ±15 điểm
```

```python
scores = np.array([85, 90, 78, 92, 88, 95, 80, 89, 91, 87])

variance = np.var(scores, ddof=1)
sd_value = np.std(scores, ddof=1)

print("Variance:", round(variance, 2))  # 27.83
print("SD:", round(sd_value, 2))        # 5.28

# SD = sqrt(Variance)
print("SD (tính từ var):", round(np.sqrt(variance), 2))
```

**Quy tắc 68-95-99.7 (cho phân phối chuẩn):**
- **68%** dữ liệu nằm trong ±1 SD
- **95%** dữ liệu nằm trong ±2 SD
- **99.7%** dữ liệu nằm trong ±3 SD

```python
# Ví dụ minh họa
rng = np.random.default_rng(42)
data = rng.normal(loc=100, scale=15, size=1000)  # IQ scores

mean_val = data.mean()
sd_val = data.std(ddof=1)

# Đếm % trong các khoảng (np.mean của mảng True/False = tỷ lệ True)
within_1sd = np.mean((data >= mean_val - sd_val) & (data <= mean_val + sd_val)) * 100
within_2sd = np.mean((data >= mean_val - 2 * sd_val) & (data <= mean_val + 2 * sd_val)) * 100
within_3sd = np.mean((data >= mean_val - 3 * sd_val) & (data <= mean_val + 3 * sd_val)) * 100

print(f"Trong ±1 SD: {within_1sd:.1f}%")  # ~68%
print(f"Trong ±2 SD: {within_2sd:.1f}%")  # ~95%
print(f"Trong ±3 SD: {within_3sd:.1f}%")  # ~99.7%
```

### 11.3.5 Hệ số biến thiên (Coefficient of Variation)

**Định nghĩa:** CV là tỷ lệ giữa độ lệch chuẩn và trung bình, biểu thị bằng %.

**Công thức:**
```
CV = (s / x̄) × 100%
```

**Tại sao dùng CV?**
- So sánh độ phân tán của các tập dữ liệu có **đơn vị khác nhau**
- So sánh độ phân tán của các tập dữ liệu có **trung bình khác nhau**

**Ví dụ:** So sánh độ biến động giữa cân nặng và chiều cao

```python
# Cân nặng (kg)
weight = np.array([60, 65, 70, 68, 72])
mean_w = weight.mean()
sd_w = weight.std(ddof=1)
cv_w = (sd_w / mean_w) * 100

print(f"Cân nặng - Mean: {mean_w} kg, SD: {sd_w:.2f}")
print(f"CV: {cv_w:.2f}%")

# Chiều cao (cm)
height = np.array([165, 170, 175, 168, 172])
mean_h = height.mean()
sd_h = height.std(ddof=1)
cv_h = (sd_h / mean_h) * 100

print(f"\nChiều cao - Mean: {mean_h} cm, SD: {sd_h:.2f}")
print(f"CV: {cv_h:.2f}%")

# So sánh
if cv_w > cv_h:
    print("\nCân nặng biến động nhiều hơn chiều cao")
else:
    print("\nChiều cao biến động nhiều hơn cân nặng")
```

**Giải thích CV:**
- CV < 10%: Biến động **thấp**
- 10% ≤ CV ≤ 20%: Biến động **trung bình**
- CV > 20%: Biến động **cao**

### 11.3.6 Tứ phân vị (Quartiles) và IQR

**Tứ phân vị** chia dữ liệu thành 4 phần bằng nhau:
- **Q1** (25%): 25% dữ liệu ≤ Q1
- **Q2** (50%): Trung vị
- **Q3** (75%): 75% dữ liệu ≤ Q3

**IQR (Interquartile Range):**
```
IQR = Q3 - Q1
```

IQR chứa **50% dữ liệu giữa**, không bị ảnh hưởng bởi outliers.

**Ví dụ:**
```
Điểm: 50, 60, 70, 75, 80, 85, 90, 95, 100
```

```python
scores = np.array([50, 60, 70, 75, 80, 85, 90, 95, 100])

# Tính tứ phân vị (np.quantile mặc định dùng cùng phương pháp với quantile() của R)
q1, q2, q3 = np.quantile(scores, [0.25, 0.5, 0.75])
print("Q1 (25%):", q1)  # 70.0
print("Q2 (50%):", q2)  # 80.0
print("Q3 (75%):", q3)  # 90.0

# Tính IQR
iqr_value = stats.iqr(scores)
print("IQR:", iqr_value)  # 20.0

# Hoặc
print("IQR (tay):", q3 - q1)
```

> **Lưu ý:** Có nhiều cách tính tứ phân vị khác nhau. `np.quantile`, `pandas.quantile` và `quantile()` của R (mặc định `type = 7`) cho cùng kết quả. Một số giáo trình/Excel dùng cách khác nên kết quả có thể lệch nhẹ.

**Phát hiện Outliers bằng IQR:**

Outliers là các giá trị:
- < Q1 - 1.5 × IQR
- > Q3 + 1.5 × IQR

```python
scores = np.array([50, 60, 70, 75, 80, 85, 90, 95, 100, 150])  # 150 là outlier?

Q1 = np.quantile(scores, 0.25)
Q3 = np.quantile(scores, 0.75)
IQR_val = stats.iqr(scores)

lower_bound = Q1 - 1.5 * IQR_val
upper_bound = Q3 + 1.5 * IQR_val

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

# Tìm outliers
outliers = scores[(scores < lower_bound) | (scores > upper_bound)]
print("Outliers:", outliers)
```

---

## 11.4 Số đo hình dạng phân phối

### 11.4.1 Độ lệch (Skewness)

**Định nghĩa:** Skewness đo **độ bất đối xứng** của phân phối.

**Công thức:**
```
Skewness = E[(X - μ)³] / σ³
```

**Giải thích:**
- **Skewness = 0**: Phân phối **đối xứng** (symmetric)
- **Skewness > 0**: Phân phối **lệch phải** (right-skewed, positive skew)
  - Đuôi dài bên phải
  - Mean > Median
- **Skewness < 0**: Phân phối **lệch trái** (left-skewed, negative skew)
  - Đuôi dài bên trái
  - Mean < Median

**Ví dụ:**

```python
from scipy.stats import skew   # Tương đương moments::skewness() trong R

rng = np.random.default_rng(42)

# 1. Phân phối đối xứng
symmetric = rng.normal(50, 10, 1000)
print("Skewness (đối xứng):", round(skew(symmetric), 3))   # ≈ 0

# 2. Phân phối lệch phải
right_skewed = np.concatenate([rng.normal(50, 5, 900), rng.normal(80, 5, 100)])
print("Skewness (lệch phải):", round(skew(right_skewed), 3))  # > 0

# 3. Phân phối lệch trái
left_skewed = np.concatenate([rng.normal(50, 5, 900), rng.normal(20, 5, 100)])
print("Skewness (lệch trái):", round(skew(left_skewed), 3))   # < 0
```

**Ví dụ thực tế:**
- **Thu nhập**: Lệch phải (ít người thu nhập rất cao)
- **Tuổi tử vong**: Lệch trái (hầu hết sống đến tuổi già)
- **Chiều cao**: Gần đối xứng

### 11.4.2 Độ nhọn (Kurtosis)

**Định nghĩa:** Kurtosis đo **độ nhọn** của phân phối, hay mức độ tập trung ở đỉnh và đuôi.

**Công thức:**
```
Kurtosis = E[(X - μ)⁴] / σ⁴
```

**Giải thích:**
- **Kurtosis = 3**: Phân phối chuẩn (mesokurtic)
- **Kurtosis > 3**: **Leptokurtic** (nhọn hơn, đuôi dày hơn)
- **Kurtosis < 3**: **Platykurtic** (tù hơn, đuôi mỏng hơn)

Thường dùng **Excess Kurtosis = Kurtosis - 3**:
- Excess = 0: Chuẩn
- Excess > 0: Nhọn hơn chuẩn
- Excess < 0: Tù hơn chuẩn

> ⚠️ **Chú ý:** `scipy.stats.kurtosis()` mặc định trả về **Excess Kurtosis** (`fisher=True`, phân phối chuẩn ≈ 0). Muốn có Kurtosis "gốc" (phân phối chuẩn ≈ 3, giống `moments::kurtosis()` của R) thì dùng `fisher=False`.

**Ví dụ:**

```python
from scipy.stats import kurtosis

rng = np.random.default_rng(42)

# 1. Phân phối chuẩn
normal_data = rng.normal(50, 10, 1000)
print("Kurtosis (chuẩn):", round(kurtosis(normal_data, fisher=False), 3))  # ≈ 3

# 2. Leptokurtic (nhọn hơn)
leptokurtic = rng.standard_t(df=3, size=1000)   # Phân phối t (R: rt)
print("Kurtosis (nhọn):", round(kurtosis(leptokurtic, fisher=False), 3))  # > 3

# 3. Platykurtic (tù hơn)
platykurtic = rng.uniform(0, 100, 1000)         # Phân phối đều (R: runif)
print("Kurtosis (tù):", round(kurtosis(platykurtic, fisher=False), 3))  # < 3
```

---

## 11.5 Bảng tần số (Frequency Table)

### 11.5.1 Bảng tần số cho dữ liệu rời rạc

**Ví dụ:** Số con trong các gia đình
```
Dữ liệu: 1, 2, 2, 3, 2, 1, 4, 2, 3, 2, 1, 2, 3, 2, 1, 2
```

```python
children = pd.Series([1, 2, 2, 3, 2, 1, 4, 2, 3, 2, 1, 2, 3, 2, 1, 2])

# Tạo bảng tần số (R: table)
freq_table = children.value_counts().sort_index()
print(freq_table)

# Tạo bảng tần suất (%) (R: prop.table)
prop_table = children.value_counts(normalize=True).sort_index() * 100
print(prop_table.round(2))

# Tạo DataFrame đẹp hơn
freq_df = pd.DataFrame({
    "So_con": freq_table.index,
    "Tan_so": freq_table.values,
    "Tan_suat": prop_table.round(2).values,
})
print(freq_df)
```

### 11.5.2 Bảng tần số cho dữ liệu liên tục

Với dữ liệu liên tục, ta cần **chia thành khoảng** (bins).

**Ví dụ:** Chiều cao của 30 sinh viên (cm)

```python
rng = np.random.default_rng(42)
heights = rng.normal(165, 8, 30)

# Phương pháp 1: Dùng pd.cut() để chia khoảng (R: cut)
breaks = np.arange(140, 191, 10)                        # 140, 150, ..., 190
height_groups = pd.cut(heights, bins=breaks, right=False)   # [140, 150), [150, 160), ...

# Tạo bảng tần số
freq_table = pd.Series(height_groups).value_counts(sort=False)
print(freq_table)

# Tạo bảng đẹp
freq_df = pd.DataFrame({
    "Khoang": freq_table.index.astype(str),
    "Tan_so": freq_table.values,
    "Tan_suat": (freq_table.values / freq_table.sum() * 100).round(2),
})
print(freq_df)

# Phương pháp 2: Tự động chia khoảng (R: hist(..., plot = FALSE))
counts, bin_edges = np.histogram(heights, bins="sturges")   # Quy tắc Sturges giống R
print("\nSố khoảng:", len(bin_edges) - 1)
print("Khoảng:", np.round(bin_edges, 2))
print("Tần số:", counts)
```

### 11.5.3 Bảng chéo (Cross-tabulation)

**Ví dụ:** Khảo sát giữa giới tính và sở thích

```python
# Tạo dữ liệu mẫu
rng = np.random.default_rng(42)
survey = pd.DataFrame({
    "gender": rng.choice(["Nam", "Nữ"], 100),
    "preference": rng.choice(["Bóng đá", "Bóng rổ", "Cầu lông"], 100),
})

# Tạo bảng chéo (R: table(gender, preference))
cross_tab = pd.crosstab(survey["gender"], survey["preference"])
print(cross_tab)

# Thêm tổng hàng và cột (R: addmargins)
print(pd.crosstab(survey["gender"], survey["preference"], margins=True, margins_name="Tổng"))

# Tính tỷ lệ theo hàng (%) (R: prop.table(..., margin = 1))
print((pd.crosstab(survey["gender"], survey["preference"], normalize="index") * 100).round(2))

# Tính tỷ lệ theo cột (%) (R: prop.table(..., margin = 2))
print((pd.crosstab(survey["gender"], survey["preference"], normalize="columns") * 100).round(2))
```

---

## 11.6 Tóm tắt thống kê với describe()

### 11.6.1 Sử dụng describe() (tương đương summary() của R)

```python
import statsmodels.api as sm
mtcars = sm.datasets.get_rdataset("mtcars").data

# Tóm tắt một biến
print(mtcars["mpg"].describe())
# count, mean, std, min, 25%, 50%, 75%, max

# Tóm tắt toàn bộ DataFrame
print(mtcars.describe().round(2))

# Tóm tắt theo nhóm (R: dplyr group_by + summarise)
summary_by_cyl = mtcars.groupby("cyl")["mpg"].agg(
    count="count",
    mean_mpg="mean",
    median_mpg="median",
    sd_mpg="std",
    min_mpg="min",
    max_mpg="max",
)
print(summary_by_cyl.round(2))
```

### 11.6.2 Mô tả chi tiết (tương đương describe() của package psych)

pandas không có sẵn hàm giống hệt `psych::describe()`, nhưng ta có thể tự viết một hàm tương tự:

```python
def describe_full(df):
    """Mô tả chi tiết: n, mean, sd, median, min, max, range, skew, kurtosis, se."""
    df = pd.DataFrame(df)
    result = pd.DataFrame({
        "n": df.count(),
        "mean": df.mean(),
        "sd": df.std(),
        "median": df.median(),
        "min": df.min(),
        "max": df.max(),
        "range": df.max() - df.min(),
        "skew": df.skew(),        # pandas: skewness đã hiệu chỉnh mẫu
        "kurtosis": df.kurt(),    # pandas: EXCESS kurtosis đã hiệu chỉnh mẫu
        "se": df.sem(),           # Sai số chuẩn
    })
    return result.round(2)

# Mô tả một biến
print(describe_full(mtcars["mpg"]))

# Mô tả nhiều biến
print(describe_full(mtcars[["mpg", "hp", "wt"]]))

# Mô tả theo nhóm (R: psych::describeBy)
for cyl, group in mtcars.groupby("cyl"):
    print(f"\n--- cyl = {cyl} ---")
    print(describe_full(group["mpg"]))
```

> Giá trị skew/kurtosis của pandas có hiệu chỉnh theo cỡ mẫu nên khác nhẹ so với `scipy.stats.skew()` (mặc định không hiệu chỉnh). Với mẫu lớn, sự khác biệt không đáng kể.

---

## 11.7 Ví dụ tổng hợp

### Ví dụ 1: Phân tích điểm thi

```python
# Tạo dữ liệu điểm thi của 50 sinh viên
rng = np.random.default_rng(123)
scores = np.round(rng.normal(75, 12, 50))
scores = np.clip(scores, 0, 100)   # Giới hạn 0-100 (R: pmax(pmin(...)))

print("=== PHÂN TÍCH ĐIỂM THI ===\n")

# 1. Số đo xu hướng trung tâm
print("1. XU HƯỚNG TRUNG TÂM")
print("Trung bình:", round(scores.mean(), 2))
print("Trung vị:", np.median(scores))
print("Mode:", get_mode(scores), "\n")

# 2. Số đo độ phân tán
print("2. ĐỘ PHÂN TÁN")
print("Range:", np.ptp(scores))
print("Variance:", round(scores.var(ddof=1), 2))
print("SD:", round(scores.std(ddof=1), 2))
print("IQR:", stats.iqr(scores))
print("CV:", round(scores.std(ddof=1) / scores.mean() * 100, 2), "%\n")

# 3. Số đo hình dạng
print("3. HÌNH DẠNG")
print("Skewness:", round(skew(scores), 3))
print("Kurtosis:", round(kurtosis(scores, fisher=False), 3), "\n")

# 4. Tứ phân vị
print("4. TỨ PHÂN VỊ")
print(pd.Series(np.quantile(scores, [0, 0.25, 0.5, 0.75, 1]),
                index=["0%", "25%", "50%", "75%", "100%"]))

# 5. Bảng tần số
print("\n5. BẢNG TẦN SỐ")
score_groups = pd.cut(scores, bins=[0, 50, 60, 70, 80, 90, 101],
                      labels=["0-50", "50-60", "60-70", "70-80", "80-90", "90-100"],
                      right=False)
print(pd.Series(score_groups).value_counts(sort=False))

# 6. Vẽ histogram
plt.hist(scores, bins=10, color="lightblue", edgecolor="white")
plt.axvline(scores.mean(), color="red", linewidth=2, linestyle="--", label="Mean")
plt.axvline(np.median(scores), color="blue", linewidth=2, linestyle="--", label="Median")
plt.title("Phân phối điểm thi")
plt.xlabel("Điểm")
plt.ylabel("Tần số")
plt.legend(loc="upper right")
plt.show()
```

### Ví dụ 2: So sánh 2 nhóm

```python
# Điểm của 2 lớp
rng = np.random.default_rng(42)
class_A = np.round(rng.normal(75, 8, 30))
class_B = np.round(rng.normal(78, 15, 30))

print("=== SO SÁNH 2 LỚP ===\n")

def summarize(x):
    return {
        "N": len(x),
        "Mean": round(x.mean(), 2),
        "Median": np.median(x),
        "SD": round(x.std(ddof=1), 2),
        "Min": x.min(),
        "Max": x.max(),
        "Range": np.ptp(x),
        "IQR": stats.iqr(x),
    }

# Tạo DataFrame so sánh
comparison = pd.DataFrame({"Class_A": summarize(class_A),
                           "Class_B": summarize(class_B)})
print(comparison)

# Vẽ boxplot so sánh
bp = plt.boxplot([class_A, class_B], tick_labels=["Lớp A", "Lớp B"], patch_artist=True)
for box, c in zip(bp["boxes"], ["lightblue", "lightcoral"]):
    box.set_facecolor(c)
plt.title("So sánh điểm 2 lớp")
plt.ylabel("Điểm")
plt.show()

# Nhận xét (dựa trên kết quả tính được)
print("\nNHẬN XÉT:")
print(f"- Lớp {'B' if class_B.mean() > class_A.mean() else 'A'} có điểm TB cao hơn")
print(f"- Lớp {'A' if class_A.std(ddof=1) < class_B.std(ddof=1) else 'B'} có độ phân tán thấp hơn (đồng đều hơn)")
print(f"- Lớp {'B' if np.ptp(class_B) > np.ptp(class_A) else 'A'} có khoảng điểm rộng hơn")
```

---

## BÀI TẬP THỰC HÀNH

### Bài tập 1: Thống kê cơ bản

Cho điểm thi của 15 sinh viên:
```
78, 85, 92, 88, 76, 90, 84, 88, 79, 91, 87, 83, 86, 89, 88
```

Tính:
1. Mean, Median, Mode
2. Variance, SD (nhớ `ddof=1`)
3. Range, IQR
4. Tứ phân vị Q1, Q2, Q3
5. Hệ số biến thiên (CV)

### Bài tập 2: Phân tích dữ liệu

Sử dụng dữ liệu `mtcars` (`sm.datasets.get_rdataset("mtcars").data`):
1. Tính thống kê mô tả cho biến `mpg`
2. So sánh `mpg` giữa các nhóm `cyl` (4, 6, 8 xy-lanh) bằng `groupby`
3. Tìm outliers trong `mpg` bằng phương pháp IQR
4. Vẽ histogram và boxplot cho `mpg`

### Bài tập 3: Bảng tần số

Cho dữ liệu tuổi của 40 người:
```python
rng = np.random.default_rng(123)
ages = rng.integers(18, 66, size=40)   # Số nguyên từ 18 đến 65 (cận trên không lấy)
```

1. Tạo bảng tần số với các khoảng: 18-25, 26-35, 36-45, 46-55, 56-65 (gợi ý: `pd.cut(ages, bins=[17, 25, 35, 45, 55, 65])`)
2. Tính tần suất (%) cho mỗi khoảng
3. Vẽ histogram

### Bài tập 4: Skewness và Kurtosis

Tạo 3 tập dữ liệu:
1. Phân phối chuẩn
2. Phân phối lệch phải (gợi ý: `rng.exponential(...)`)
3. Phân phối lệch trái

Tính và so sánh Skewness, Kurtosis của 3 tập.

### Bài tập 5: Ứng dụng thực tế

Thu thập dữ liệu chiều cao của bạn bè (ít nhất 20 người):
1. Tính các thống kê mô tả
2. Vẽ histogram và boxplot
3. Nhận xét về phân phối
4. Tìm outliers (nếu có)

---

## CÂU HỎI ÔN TẬP

1. Phân biệt Mean, Median và Mode? Khi nào nên dùng mỗi loại?
2. Giải thích ý nghĩa của Standard Deviation?
3. Tại sao chia cho (n-1) thay vì n khi tính variance mẫu? Trong NumPy, tham số nào điều khiển điều này?
4. IQR đo lường gì? Tại sao IQR tốt hơn Range?
5. Skewness dương nghĩa là gì? Cho ví dụ?
6. Phân biệt Variance và Standard Deviation?
7. Hệ số biến thiên (CV) dùng để làm gì?
8. Làm thế nào để phát hiện outliers?

---

## TÀI LIỆU THAM KHẢO

1. **NumPy**: [`numpy.mean`, `numpy.median`, `numpy.std`, `numpy.quantile`](https://numpy.org/doc/stable/reference/routines.statistics.html)
2. **pandas**: [`DataFrame.describe`, `groupby`, `crosstab`](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics)
3. **SciPy**: [`scipy.stats.skew`, `kurtosis`, `iqr`](https://docs.scipy.org/doc/scipy/reference/stats.html)
4. **Python Standard Library**: [`statistics` module](https://docs.python.org/3/library/statistics.html)

---

## TỔNG KẾT

### Bảng công thức tổng hợp:

| Chỉ số | Công thức | Hàm Python | Hàm R | Ý nghĩa |
|--------|-----------|-------|-------|---------|
| **Mean** | Σx / n | `np.mean()`, `s.mean()` | `mean()` | Giá trị trung bình |
| **Median** | Giá trị giữa | `np.median()`, `s.median()` | `median()` | Giá trị chính giữa |
| **Mode** | Xuất hiện nhiều nhất | `s.mode()`, `statistics.multimode()` | (tự viết) | Giá trị phổ biến |
| **Variance** | Σ(x-x̄)²/(n-1) | `np.var(x, ddof=1)`, `s.var()` | `var()` | Độ phân tán |
| **SD** | √Variance | `np.std(x, ddof=1)`, `s.std()` | `sd()` | Độ lệch trung bình |
| **Range** | Max - Min | `np.ptp()` | `diff(range())` | Khoảng biến thiên |
| **IQR** | Q3 - Q1 | `scipy.stats.iqr()` | `IQR()` | Khoảng tứ phân vị |
| **CV** | (SD/Mean)×100% | Tự tính | Tự tính | Hệ số biến thiên |
| **Skewness** | E[(X-μ)³]/σ³ | `scipy.stats.skew()` | `moments::skewness()` | Độ lệch |
| **Kurtosis** | E[(X-μ)⁴]/σ⁴ | `scipy.stats.kurtosis(x, fisher=False)` | `moments::kurtosis()` | Độ nhọn |

### Lưu ý quan trọng:

✅ **Mean**: Nhạy cảm với outliers
✅ **Median**: Tốt khi có outliers
✅ **SD**: Cùng đơn vị với dữ liệu gốc – **nhớ `ddof=1` với NumPy**
✅ **IQR**: Không bị ảnh hưởng outliers
✅ **CV**: So sánh độ phân tán giữa các tập dữ liệu khác nhau
✅ **Skewness**: Đo độ bất đối xứng
✅ **Kurtosis**: Đo độ nhọn – `scipy` mặc định trả về *excess kurtosis*

### Quy trình phân tích:

1. **Khám phá**: `df.describe()`, `df.info()`, `df.head()`
2. **Xu hướng trung tâm**: `mean()`, `median()`, `mode()`
3. **Độ phân tán**: `std()`, `var()`, `np.ptp()`, `stats.iqr()`
4. **Hình dạng**: `stats.skew()`, `stats.kurtosis()`
5. **Trực quan hóa**: `plt.hist()`, `plt.boxplot()`
6. **Outliers**: Phương pháp IQR
7. **Báo cáo**: Tóm tắt kết quả

---

**Cập nhật**: Tháng 3/2026
