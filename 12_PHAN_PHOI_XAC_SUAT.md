# BÀI 12: PHÂN PHỐI XÁC SUẤT

## Mục tiêu học tập
- Hiểu khái niệm biến ngẫu nhiên và phân phối xác suất
- Phân biệt biến ngẫu nhiên rời rạc và liên tục
- Nắm vững các phân phối xác suất quan trọng (Binomial, Poisson, Normal)
- Tính xác suất bằng các hàm phân phối trong Python (`scipy.stats`)
- Hiểu và áp dụng phân phối chuẩn (Normal Distribution)
- Sử dụng Z-score và bảng Z
- Áp dụng phân phối xác suất vào bài toán thực tế

> **Chuẩn bị:**
> ```python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy import stats
> ```

---

## 12.1 Biến ngẫu nhiên

### 12.1.1 Biến ngẫu nhiên là gì?

**Biến ngẫu nhiên** (Random Variable) là một biến có giá trị được xác định bởi **kết quả của một sự kiện ngẫu nhiên**.

**Ký hiệu:** Thường dùng chữ in hoa X, Y, Z

**Ví dụ:**
- X = Số chấm khi tung xúc xắc (X có thể = 1, 2, 3, 4, 5, 6)
- Y = Chiều cao của sinh viên (Y có thể = 165cm, 170cm, ...)
- Z = Số cuộc gọi đến tổng đài trong 1 giờ

### 12.1.2 Phân loại biến ngẫu nhiên

**1. Biến ngẫu nhiên rời rạc (Discrete Random Variable)**
- Nhận **số hữu hạn** hoặc **đếm được** các giá trị
- Thường là **số đếm**

**Ví dụ:**
- Số con trong gia đình: 0, 1, 2, 3, ...
- Số lần tung đồng xu đến khi ra mặt ngửa
- Số khách hàng đến cửa hàng trong 1 ngày

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Ví dụ: Tung xúc xắc 10 lần
rng = np.random.default_rng(42)            # R: set.seed(42)
dice_rolls = rng.integers(1, 7, size=10)   # Số nguyên từ 1 đến 6 (cận trên 7 không lấy)
print("Kết quả tung xúc xắc:", dice_rolls)
print("Các giá trị khác nhau:", np.unique(dice_rolls))
```

**2. Biến ngẫu nhiên liên tục (Continuous Random Variable)**
- Nhận **vô số** giá trị trong một khoảng
- Thường là **phép đo**

**Ví dụ:**
- Chiều cao: 165.5cm, 170.2cm, ...
- Thời gian chờ xe bus: 2.5 phút, 5.8 phút, ...
- Nhiệt độ: 25.3°C, 30.7°C, ...

```python
# Ví dụ: Chiều cao ngẫu nhiên
rng = np.random.default_rng(42)
heights = rng.normal(loc=170, scale=5, size=10)
print("Chiều cao (cm):", np.round(heights, 2))
```

### 12.1.3 Hàm phân phối xác suất

**Hàm khối xác suất (PMF)** - cho biến rời rạc
- P(X = x): Xác suất X nhận giá trị x
- Tổng tất cả xác suất = 1

**Hàm mật độ xác suất (PDF)** - cho biến liên tục
- f(x): Mật độ xác suất tại x
- Diện tích dưới đường cong = 1

**Hàm phân phối tích lũy (CDF)**
- F(x) = P(X ≤ x): Xác suất X nhỏ hơn hoặc bằng x

### 12.1.4 Quy ước hàm trong `scipy.stats` (đối chiếu với R)

Mỗi phân phối trong `scipy.stats` là một **đối tượng** (`stats.binom`, `stats.poisson`, `stats.norm`, ...) có các phương thức giống nhau:

| Ý nghĩa | R (tiền tố) | `scipy.stats` |
|---------|-------------|---------------|
| Khối xác suất P(X = x) (rời rạc) | `d` (`dbinom`) | `.pmf(x, ...)` |
| Mật độ f(x) (liên tục) | `d` (`dnorm`) | `.pdf(x, ...)` |
| Tích lũy P(X ≤ x) | `p` (`pbinom`) | `.cdf(x, ...)` |
| P(X > x) = 1 − CDF | `1 - p...` | `.sf(x, ...)` (*survival function*) |
| Phân vị (tìm x khi biết xác suất) | `q` (`qbinom`) | `.ppf(p, ...)` |
| Sinh số ngẫu nhiên | `r` (`rbinom`) | `.rvs(..., size=n)` |

---

## 12.2 Phân phối Binomial (Nhị thức)

### 12.2.1 Định nghĩa

**Phân phối Binomial** mô tả số lần **thành công** trong **n lần thử độc lập**, mỗi lần có xác suất thành công là **p**.

**Điều kiện áp dụng:**
1. Có **n lần thử** độc lập
2. Mỗi lần chỉ có **2 kết quả**: thành công hoặc thất bại
3. Xác suất thành công **p** giống nhau ở mỗi lần

**Ký hiệu:** X ~ B(n, p)
- n: số lần thử
- p: xác suất thành công

**Công thức:**
```
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
```

### 12.2.2 Ví dụ cơ bản

**Ví dụ 1:** Tung đồng xu 10 lần, xác suất ra mặt ngửa = 0.5

```python
# Xác suất có đúng 5 lần ngửa
n = 10
p = 0.5
k = 5

prob = stats.binom.pmf(k, n, p)     # R: dbinom(k, size = n, prob = p)
print("P(X = 5) =", round(prob, 4))
# 0.2461
```

**Ví dụ 2:** Bắn cung 20 lần, xác suất trúng đích = 0.7

```python
# Xác suất trúng đích đúng 15 lần
prob_15 = stats.binom.pmf(15, 20, 0.7)
print("P(X = 15) =", round(prob_15, 4))

# Xác suất trúng ít nhất 15 lần
prob_at_least_15 = stats.binom.pmf(np.arange(15, 21), 20, 0.7).sum()
print("P(X ≥ 15) =", round(prob_at_least_15, 4))

# Cách khác: P(X ≥ 15) = P(X > 14) = sf(14)
print("P(X ≥ 15) =", round(stats.binom.sf(14, 20, 0.7), 4))
```

### 12.2.3 Các hàm trong scipy.stats

`scipy.stats.binom` có 4 phương thức chính:

**1. `binom.pmf(k, n, p)`** - Hàm khối xác suất (PMF)
- Tính P(X = k) – R: `dbinom`

**2. `binom.cdf(k, n, p)`** - Hàm phân phối tích lũy (CDF)
- Tính P(X ≤ k) – R: `pbinom`

**3. `binom.ppf(q, n, p)`** - Hàm phân vị (Quantile)
- Tìm k nhỏ nhất sao cho P(X ≤ k) ≥ q – R: `qbinom`

**4. `binom.rvs(n, p, size=m)`** - Sinh số ngẫu nhiên
- Sinh m số ngẫu nhiên từ phân phối Binomial – R: `rbinom`

```python
n = 10
p = 0.5

print("=== CÁC HÀM BINOMIAL ===\n")

# 1. pmf - P(X = k)
print("1. P(X = 5) =", stats.binom.pmf(5, n, p))

# 2. cdf - P(X ≤ k)
print("2. P(X ≤ 5) =", stats.binom.cdf(5, n, p))

# 3. ppf - Tìm k sao cho P(X ≤ k) = 0.5
print("3. Median =", stats.binom.ppf(0.5, n, p))

# 4. rvs - Sinh 10 số ngẫu nhiên
random_values = stats.binom.rvs(n, p, size=10, random_state=42)
print("4. Random:", random_values)
```

### 12.2.4 Mean và Variance của Binomial

**Kỳ vọng (Mean):**
```
E(X) = n × p
```

**Phương sai (Variance):**
```
Var(X) = n × p × (1 - p)
```

```python
n = 100
p = 0.3

mean_val = n * p
var_val = n * p * (1 - p)
sd_val = np.sqrt(var_val)

print("Mean:", mean_val)          # 30.0
print("Variance:", var_val)       # 21.0
print("SD:", round(sd_val, 2))    # 4.58

# scipy tính sẵn cho ta
print(stats.binom.mean(n, p), stats.binom.var(n, p), round(stats.binom.std(n, p), 2))
```

### 12.2.5 Ví dụ thực tế

**Ví dụ:** Một công ty có tỷ lệ sản phẩm lỗi là 5%. Kiểm tra 50 sản phẩm.

**a) Xác suất có đúng 3 sản phẩm lỗi?**
```python
n = 50
p = 0.05
prob_3 = stats.binom.pmf(3, n, p)
print("P(X = 3) =", round(prob_3, 4))
```

**b) Xác suất có nhiều hơn 5 sản phẩm lỗi?**
```python
prob_more_5 = 1 - stats.binom.cdf(5, n, p)   # hoặc stats.binom.sf(5, n, p)
print("P(X > 5) =", round(prob_more_5, 4))
```

**c) Số sản phẩm lỗi kỳ vọng?**
```python
expected = n * p
print("Kỳ vọng:", expected, "sản phẩm")
```

### 12.2.6 Vẽ biểu đồ Binomial

```python
n = 20
p = 0.3

# Tạo dữ liệu
x = np.arange(0, n + 1)
pmf = stats.binom.pmf(x, n, p)

# Vẽ biểu đồ
plt.bar(x, pmf, color="steelblue")
plt.title(f"Binomial Distribution (n = {n}, p = {p})")
plt.xlabel("Số lần thành công")
plt.ylabel("Xác suất")

# Thêm đường trung bình
plt.axvline(n * p, color="red", linewidth=2, linestyle="--", label="Mean")
plt.legend(loc="upper right")
plt.show()
```

> Khác với `barplot()` của R, `plt.bar(x, ...)` đặt cột tại đúng tọa độ x nên đường trung bình vẽ tại `n*p` (không cần cộng 0.5).

---

## 12.3 Phân phối Poisson

### 12.3.1 Định nghĩa

**Phân phối Poisson** mô tả số lần một sự kiện xảy ra trong một **khoảng thời gian/không gian cố định**.

**Điều kiện áp dụng:**
1. Sự kiện xảy ra **độc lập**
2. Tỷ lệ trung bình (λ) **không đổi**
3. Hai sự kiện **không xảy ra đồng thời**

**Ký hiệu:** X ~ Poisson(λ)
- λ (lambda): Số sự kiện trung bình

**Công thức:**
```
P(X = k) = (e^(-λ) × λ^k) / k!
```

### 12.3.2 Ví dụ cơ bản

**Ví dụ 1:** Số cuộc gọi đến tổng đài trung bình 3 cuộc/phút

```python
lam = 3   # Không đặt tên biến là "lambda" vì đây là từ khóa của Python!

# Xác suất có đúng 5 cuộc gọi
prob_5 = stats.poisson.pmf(5, lam)
print("P(X = 5) =", round(prob_5, 4))

# Xác suất có nhiều hơn 5 cuộc gọi
prob_more_5 = 1 - stats.poisson.cdf(5, lam)
print("P(X > 5) =", round(prob_more_5, 4))
```

### 12.3.3 Các hàm trong scipy.stats

**1. `poisson.pmf(k, mu)`** - Hàm khối xác suất – R: `dpois`
- Tính P(X = k)

**2. `poisson.cdf(k, mu)`** - Hàm phân phối tích lũy – R: `ppois`
- Tính P(X ≤ k)

**3. `poisson.ppf(q, mu)`** - Hàm phân vị – R: `qpois`
- Tìm k sao cho P(X ≤ k) = q

**4. `poisson.rvs(mu, size=m)`** - Sinh số ngẫu nhiên – R: `rpois`

```python
lam = 4

print("=== CÁC HÀM POISSON ===\n")

# 1. pmf
print("1. P(X = 4) =", stats.poisson.pmf(4, lam))

# 2. cdf
print("2. P(X ≤ 4) =", stats.poisson.cdf(4, lam))

# 3. ppf
print("3. Median =", stats.poisson.ppf(0.5, lam))

# 4. rvs
random_values = stats.poisson.rvs(lam, size=10, random_state=42)
print("4. Random:", random_values)
```

### 12.3.4 Mean và Variance của Poisson

**Đặc điểm quan trọng:** Mean = Variance = λ

```python
lam = 5

print("Mean:", stats.poisson.mean(lam))
print("Variance:", stats.poisson.var(lam))
print("SD:", np.sqrt(lam))
```

### 12.3.5 Ví dụ thực tế

**Ví dụ:** Một trang web nhận trung bình 100 lượt truy cập/giờ.

**a) Xác suất có đúng 90 lượt trong 1 giờ?**
```python
lam = 100
prob_90 = stats.poisson.pmf(90, lam)
print("P(X = 90) =", round(prob_90, 4))
```

**b) Xác suất có ít hơn 80 lượt?**
```python
prob_less_80 = stats.poisson.cdf(79, lam)   # P(X < 80) = P(X ≤ 79)
print("P(X < 80) =", round(prob_less_80, 4))
```

**c) Xác suất có từ 90 đến 110 lượt?**
```python
prob_90_110 = stats.poisson.cdf(110, lam) - stats.poisson.cdf(89, lam)
print("P(90 ≤ X ≤ 110) =", round(prob_90_110, 4))
```

### 12.3.6 Vẽ biểu đồ Poisson

```python
lam = 5

# Tạo dữ liệu
x = np.arange(0, 16)
pmf = stats.poisson.pmf(x, lam)

# Vẽ biểu đồ
plt.bar(x, pmf, color="coral")
plt.title(f"Poisson Distribution (λ = {lam})")
plt.xlabel("Số sự kiện")
plt.ylabel("Xác suất")

# Thêm đường trung bình
plt.axvline(lam, color="red", linewidth=2, linestyle="--")
plt.show()
```

---

## 12.4 Phân phối Chuẩn (Normal Distribution)

### 12.4.1 Định nghĩa

**Phân phối chuẩn** (hay phân phối Gauss) là phân phối liên tục quan trọng nhất trong thống kê.

**Ký hiệu:** X ~ N(μ, σ²)
- μ (mu): Trung bình
- σ² (sigma²): Phương sai
- σ: Độ lệch chuẩn

**Đặc điểm:**
1. Hình dạng **chuông** (bell-shaped)
2. **Đối xứng** qua μ
3. Mean = Median = Mode = μ
4. Diện tích dưới đường cong = 1

### 12.4.2 Phân phối chuẩn chuẩn hóa

**Phân phối chuẩn chuẩn hóa:** Z ~ N(0, 1)
- Mean = 0
- SD = 1

**Công thức chuyển đổi (Z-score):**
```
Z = (X - μ) / σ
```

Z-score cho biết X cách trung bình bao nhiêu độ lệch chuẩn.

```python
# Ví dụ: IQ có mean = 100, SD = 15
# IQ = 130 tương ứng Z-score bao nhiêu?

x = 130
mu = 100
sigma = 15

z = (x - mu) / sigma
print("Z-score:", z)  # 2.0

# Giải thích: IQ 130 cao hơn trung bình 2 độ lệch chuẩn
```

### 12.4.3 Các hàm trong scipy.stats

Trong `scipy.stats.norm`: **`loc` = trung bình μ**, **`scale` = độ lệch chuẩn σ** (không phải phương sai!).

**1. `norm.pdf(x, loc, scale)`** - Hàm mật độ xác suất (PDF) – R: `dnorm`
- Tính mật độ tại x

**2. `norm.cdf(x, loc, scale)`** - Hàm phân phối tích lũy (CDF) – R: `pnorm`
- Tính P(X ≤ x)

**3. `norm.ppf(q, loc, scale)`** - Hàm phân vị – R: `qnorm`
- Tìm x sao cho P(X ≤ x) = q

**4. `norm.rvs(loc, scale, size=n)`** - Sinh số ngẫu nhiên – R: `rnorm`

```python
mu = 100
sigma = 15

print("=== CÁC HÀM NORMAL ===\n")

# 1. pdf - Mật độ
print("1. Mật độ tại x=100:", stats.norm.pdf(100, mu, sigma))

# 2. cdf - P(X ≤ x)
print("2. P(X ≤ 100) =", stats.norm.cdf(100, mu, sigma))  # 0.5

# 3. ppf - Tìm x
print("3. Giá trị tại 95%:", stats.norm.ppf(0.95, mu, sigma))

# 4. rvs - Sinh số ngẫu nhiên
random_values = stats.norm.rvs(mu, sigma, size=5, random_state=42)
print("4. Random:", np.round(random_values, 2))
```

> **Mẹo:** Có thể "cố định" tham số để tạo một phân phối cụ thể rồi dùng lại: `iq = stats.norm(loc=100, scale=15)` → `iq.cdf(130)`, `iq.ppf(0.95)`, `iq.rvs(5)`.

### 12.4.4 Quy tắc 68-95-99.7

Trong phân phối chuẩn:
- **68%** dữ liệu nằm trong μ ± 1σ
- **95%** dữ liệu nằm trong μ ± 2σ
- **99.7%** dữ liệu nằm trong μ ± 3σ

```python
mu = 100
sigma = 15
iq = stats.norm(loc=mu, scale=sigma)

# Tính xác suất trong các khoảng
prob_1sd = iq.cdf(mu + sigma) - iq.cdf(mu - sigma)
prob_2sd = iq.cdf(mu + 2 * sigma) - iq.cdf(mu - 2 * sigma)
prob_3sd = iq.cdf(mu + 3 * sigma) - iq.cdf(mu - 3 * sigma)

print("=== QUY TẮC 68-95-99.7 ===")
print("P(μ ± 1σ) =", round(prob_1sd, 4))  # 0.6827
print("P(μ ± 2σ) =", round(prob_2sd, 4))  # 0.9545
print("P(μ ± 3σ) =", round(prob_3sd, 4))  # 0.9973
```

### 12.4.5 Ví dụ tính xác suất

**Ví dụ:** Điểm thi có phân phối chuẩn với μ = 70, σ = 10

**a) Xác suất sinh viên đạt trên 80 điểm?**
```python
mu = 70
sigma = 10

# P(X > 80)
prob_above_80 = 1 - stats.norm.cdf(80, mu, sigma)
print("P(X > 80) =", round(prob_above_80, 4))
```

**b) Xác suất đạt từ 60-80 điểm?**
```python
# P(60 ≤ X ≤ 80)
prob_60_80 = stats.norm.cdf(80, mu, sigma) - stats.norm.cdf(60, mu, sigma)
print("P(60 ≤ X ≤ 80) =", round(prob_60_80, 4))
```

**c) Điểm tối thiểu để vào top 10%?**
```python
# Tìm x sao cho P(X ≥ x) = 0.1
# Tức P(X ≤ x) = 0.9
cutoff = stats.norm.ppf(0.9, mu, sigma)
print("Điểm tối thiểu:", round(cutoff, 2))
```

### 12.4.6 Sử dụng Z-score

```python
# Ví dụ: IQ ~ N(100, 15²)
mu = 100
sigma = 15

# Câu hỏi: Xác suất IQ > 130?

# Cách 1: Dùng trực tiếp
prob1 = 1 - stats.norm.cdf(130, mu, sigma)

# Cách 2: Chuyển sang Z-score (phân phối chuẩn tắc: loc=0, scale=1 là mặc định)
z = (130 - mu) / sigma
prob2 = 1 - stats.norm.cdf(z)

print("Cách 1:", round(prob1, 4))
print("Cách 2:", round(prob2, 4))
print("Hai cách cho kết quả giống nhau!")
```

### 12.4.7 Vẽ đường cong chuẩn

```python
mu = 100
sigma = 15

# Tạo dữ liệu
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
y = stats.norm.pdf(x, mu, sigma)

# Vẽ đường cong
plt.plot(x, y, color="blue", linewidth=2)
plt.title("Phân phối chuẩn N(100, 15²)")
plt.xlabel("Giá trị")
plt.ylabel("Mật độ")

# Tô màu vùng dưới đường cong (ví dụ: P(X ≤ 100)) – R: polygon
plt.fill_between(x, y, where=(x <= mu), color="blue", alpha=0.3)

# Thêm đường mean
plt.axvline(mu, color="red", linewidth=2, linestyle="--")
plt.text(mu + 2, y.max() * 0.9, "μ = 100", color="red")
plt.show()
```

### 12.4.8 Kiểm tra tính chuẩn (Normality Test)

**Q-Q Plot** (Quantile-Quantile Plot) - Kiểm tra trực quan

```python
# Tạo dữ liệu chuẩn
rng = np.random.default_rng(42)
data_normal = rng.normal(50, 10, 100)

# Q-Q plot (R: qqnorm + qqline)
stats.probplot(data_normal, dist="norm", plot=plt)
plt.title("Q-Q Plot - Dữ liệu chuẩn")
plt.show()

# Nếu các điểm nằm gần đường thẳng → dữ liệu có phân phối chuẩn
```

**Shapiro-Wilk Test** - Kiểm tra thống kê

```python
# H0: Dữ liệu có phân phối chuẩn
# H1: Dữ liệu không có phân phối chuẩn

# Test với dữ liệu chuẩn (R: shapiro.test)
test_result = stats.shapiro(data_normal)
print("W =", round(test_result.statistic, 4))
print("P-value:", round(test_result.pvalue, 4))

# Nếu p-value > 0.05 → Không bác bỏ H0 → Chưa có bằng chứng dữ liệu KHÔNG chuẩn
if test_result.pvalue > 0.05:
    print("Dữ liệu có phân phối chuẩn")
else:
    print("Dữ liệu KHÔNG có phân phối chuẩn")
```

---

## 12.5 Định lý giới hạn trung tâm (Central Limit Theorem)

### 12.5.1 Định lý

**Định lý giới hạn trung tâm (CLT)** phát biểu:

Khi lấy mẫu đủ lớn (n ≥ 30) từ một tổng thể **bất kỳ**, phân phối của **trung bình mẫu** sẽ tiến đến phân phối chuẩn.

```
X̄ ~ N(μ, σ²/n)
```

Trong đó:
- μ: Trung bình tổng thể
- σ²: Phương sai tổng thể
- n: Kích thước mẫu

### 12.5.2 Minh họa CLT

```python
# Minh họa CLT với phân phối đều (không chuẩn)
rng = np.random.default_rng(42)

n_samples = 1000
sample_size = 30

# Tạo phân phối gốc (đều từ 0-10)
population = rng.uniform(0, 10, 10000)

# Lấy mẫu và tính trung bình (R: replicate)
sample_means = np.array([
    rng.choice(population, sample_size, replace=False).mean()
    for _ in range(n_samples)
])

# Vẽ histogram của trung bình mẫu
plt.hist(sample_means, bins=30, density=True, color="lightblue",
         edgecolor="white", label="Histogram")

# Thêm đường cong chuẩn lý thuyết (R: curve(dnorm(...), add = TRUE))
xs = np.linspace(sample_means.min(), sample_means.max(), 200)
plt.plot(xs, stats.norm.pdf(xs, population.mean(), population.std(ddof=1) / np.sqrt(sample_size)),
         color="red", linewidth=2, label="Phân phối chuẩn lý thuyết")

plt.title("Phân phối của trung bình mẫu (CLT)")
plt.xlabel("Trung bình mẫu")
plt.legend(loc="upper right")
plt.show()
```

### 12.5.3 Ứng dụng CLT

**Ví dụ:** Chiều cao sinh viên có μ = 165cm, σ = 8cm. Lấy mẫu 40 sinh viên.

**Xác suất trung bình mẫu > 167cm?**

```python
mu = 165
sigma = 8
n = 40

# Theo CLT: X̄ ~ N(165, 8²/40)
mu_xbar = mu
sigma_xbar = sigma / np.sqrt(n)

# P(X̄ > 167)
prob = 1 - stats.norm.cdf(167, mu_xbar, sigma_xbar)
print("P(X̄ > 167) =", round(prob, 4))
```

---

## 12.6 Các phân phối khác

### 12.6.1 Phân phối Uniform (Đều)

**Mọi giá trị** trong khoảng [a, b] có **xác suất như nhau**.

```python
# Uniform(0, 1): trong scipy, uniform(loc=a, scale=b-a)
x = np.linspace(0, 1, 101)
plt.plot(x, stats.uniform.pdf(x, loc=0, scale=1))
plt.title("Uniform Distribution (0, 1)")
plt.ylabel("Density")
plt.show()

# Sinh số ngẫu nhiên (R: runif(10, 0, 1))
rng = np.random.default_rng(1)
random_uniform = rng.uniform(0, 1, 10)
print("Random:", np.round(random_uniform, 3))
```

### 12.6.2 Phân phối Exponential (Mũ)

Mô tả **thời gian chờ** giữa các sự kiện trong Poisson.

> ⚠️ **Chú ý tham số:** R dùng `rate = λ`, còn `scipy.stats.expon` dùng **`scale = 1/λ`**.

```python
# Exponential(lambda = 2)
lam = 2

x = np.linspace(0, 5, 501)
plt.plot(x, stats.expon.pdf(x, scale=1 / lam))   # R: dexp(x, rate = 2)
plt.title("Exponential Distribution (λ = 2)")
plt.ylabel("Density")
plt.show()

# Thời gian chờ trung bình = 1/lambda
# Ví dụ: xe bus trung bình 6 chuyến/giờ (λ = 6) → chờ trung bình 1/6 giờ = 10 phút
mean_wait = 1 / lam
print("Thời gian chờ TB:", mean_wait, "đơn vị")
```

---

## 12.7 Tổng hợp và so sánh

### 12.7.1 Bảng so sánh các phân phối

| Phân phối | Loại | Tham số | Ứng dụng | `scipy.stats` | R |
|-----------|------|---------|----------|---------------|---|
| **Binomial** | Rời rạc | n, p | Số lần thành công | `binom.pmf(k, n, p)` | `dbinom()` |
| **Poisson** | Rời rạc | λ | Số sự kiện/thời gian | `poisson.pmf(k, mu)` | `dpois()` |
| **Normal** | Liên tục | μ, σ | Nhiều hiện tượng tự nhiên | `norm.pdf(x, loc, scale)` | `dnorm()` |
| **Uniform** | Liên tục | a, b | Số ngẫu nhiên | `uniform.pdf(x, loc=a, scale=b-a)` | `dunif()` |
| **Exponential** | Liên tục | λ | Thời gian chờ | `expon.pdf(x, scale=1/λ)` | `dexp()` |

### 12.7.2 Khi nào dùng phân phối nào?

**Binomial:**
- Đếm số lần thành công
- Có số lần thử cố định
- Ví dụ: Tung đồng xu, khảo sát có/không

**Poisson:**
- Đếm số sự kiện hiếm
- Trong khoảng thời gian/không gian
- Ví dụ: Số cuộc gọi, số lỗi, số tai nạn

**Normal:**
- Dữ liệu liên tục, đối xứng
- Nhiều hiện tượng tự nhiên
- Ví dụ: Chiều cao, điểm thi, IQ

---

## BÀI TẬP THỰC HÀNH

### Bài tập 1: Binomial

Một học sinh làm bài trắc nghiệm 20 câu, mỗi câu 4 đáp án. Học sinh đoán ngẫu nhiên.

1. Xác suất trả lời đúng 5 câu?
2. Xác suất trả lời đúng ít nhất 10 câu?
3. Số câu đúng kỳ vọng?

### Bài tập 2: Poisson

Một cửa hàng nhận trung bình 12 khách hàng/giờ.

1. Xác suất có đúng 10 khách trong 1 giờ?
2. Xác suất có nhiều hơn 15 khách?
3. Xác suất có từ 10-15 khách?

### Bài tập 3: Normal

Điểm thi có phân phối N(75, 10²).

1. Xác suất sinh viên đạt trên 85 điểm?
2. Xác suất đạt từ 65-85 điểm?
3. Điểm tối thiểu để vào top 20%?
4. Tính Z-score của điểm 90

### Bài tập 4: CLT

Thời gian làm bài có μ = 45 phút, σ = 8 phút. Lấy mẫu 36 sinh viên.

1. Xác suất trung bình thời gian > 47 phút?
2. Xác suất trung bình thời gian < 43 phút?

### Bài tập 5: So sánh phân phối

Vẽ và so sánh (gợi ý: dùng `plt.subplots(1, 3)`):
1. Binomial(20, 0.3) vs Binomial(20, 0.7)
2. Poisson(3) vs Poisson(10)
3. Normal(0,1) vs Normal(0,4) (*lưu ý: N(0, 4) nghĩa là phương sai 4 → `scale=2`*)

---

## CÂU HỎI ÔN TẬP

1. Phân biệt biến ngẫu nhiên rời rạc và liên tục?
2. Khi nào dùng Binomial? Khi nào dùng Poisson?
3. Giải thích quy tắc 68-95-99.7?
4. Z-score là gì? Cách tính?
5. Định lý giới hạn trung tâm nói gì?
6. Phân phối nào có Mean = Variance?
7. Tại sao phân phối chuẩn quan trọng?

---

## TÀI LIỆU THAM KHẢO

1. **SciPy Documentation**: [Statistical functions (`scipy.stats`)](https://docs.scipy.org/doc/scipy/reference/stats.html)
2. **SciPy Tutorial – Probability distributions**: https://docs.scipy.org/doc/scipy/tutorial/stats.html
3. **Central Limit Theorem**: Khan Academy

---

## TỔNG KẾT

### Công thức tổng hợp:

**Binomial:**
- P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
- E(X) = np, Var(X) = np(1-p)

**Poisson:**
- P(X = k) = (e^(-λ) × λ^k) / k!
- E(X) = Var(X) = λ

**Normal:**
- X ~ N(μ, σ²)
- Z = (X - μ) / σ
- Quy tắc 68-95-99.7

### Phương thức quan trọng của `scipy.stats`:

| Phương thức | Ý nghĩa | Tiền tố R |
|---------|---------|---------|
| **`.pmf()` / `.pdf()`** | Mass / Density (PMF/PDF) | `d` |
| **`.cdf()`** | Probability (CDF) | `p` |
| **`.sf()`** | 1 − CDF (xác suất đuôi phải) | `1 - p` |
| **`.ppf()`** | Quantile | `q` |
| **`.rvs()`** | Random | `r` |

### Lưu ý:

✅ Binomial: n lần thử, 2 kết quả
✅ Poisson: Sự kiện hiếm, λ = mean
✅ Normal: Đối xứng, quy tắc 68-95-99.7 – `scale` là **độ lệch chuẩn**
✅ CLT: n ≥ 30 → X̄ ~ Normal
✅ Z-score: Chuẩn hóa dữ liệu

---

**Cập nhật**: Tháng 3/2026
