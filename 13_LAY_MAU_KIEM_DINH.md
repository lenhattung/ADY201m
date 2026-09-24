# BÀI 13: LẤY MẪU VÀ KIỂM ĐỊNH THỐNG KÊ

## Mục tiêu học tập
- Hiểu khái niệm tổng thể và mẫu
- Nắm vững các phương pháp lấy mẫu
- Hiểu sai số chuẩn và khoảng tin cậy
- Nắm vững các bước kiểm định giả thuyết
- Thực hiện kiểm định t-test (một mẫu, hai mẫu)
- Thực hiện kiểm định chi-square
- Hiểu p-value và mức ý nghĩa α
- Áp dụng kiểm định vào bài toán thực tế

> **Chuẩn bị:**
> ```python
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> from scipy import stats
> ```
> Các hàm kiểm định dùng trong bài nằm trong `scipy.stats` (yêu cầu SciPy ≥ 1.11 để dùng `.confidence_interval()`) và `statsmodels`.

---

## 13.1 Tổng thể và Mẫu

### 13.1.1 Định nghĩa

**Tổng thể (Population)**
- Tập hợp **tất cả** các đối tượng quan tâm
- Ký hiệu tham số: μ (mean), σ (SD), p (tỷ lệ)

**Mẫu (Sample)**
- Một **tập con** của tổng thể
- Ký hiệu thống kê: x̄ (mean), s (SD), p̂ (tỷ lệ)

**Ví dụ:**
- Tổng thể: Tất cả sinh viên Việt Nam (hàng triệu người)
- Mẫu: 1000 sinh viên được khảo sát

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Minh họa
rng = np.random.default_rng(42)

# Tổng thể (giả sử)
population = rng.normal(loc=170, scale=8, size=100_000)
print(f"Tổng thể - μ: {population.mean():.2f}  σ: {population.std():.2f}")

# Lấy mẫu (R: sample(population, 100))
sample_data = rng.choice(population, size=100, replace=False)
print(f"Mẫu - x̄: {sample_data.mean():.2f}  s: {sample_data.std(ddof=1):.2f}")
```

### 13.1.2 Tại sao cần lấy mẫu?

**Lý do:**
1. **Chi phí**: Khảo sát toàn bộ tổng thể rất tốn kém
2. **Thời gian**: Mất quá nhiều thời gian
3. **Không khả thi**: Một số tổng thể vô hạn hoặc không tiếp cận được
4. **Phá hủy**: Kiểm tra tuổi thọ bóng đèn → phải phá hủy

**Mục tiêu:** Từ mẫu, **ước lượng** tham số của tổng thể.

---

## 13.2 Các phương pháp lấy mẫu

### 13.2.1 Lấy mẫu ngẫu nhiên đơn giản (Simple Random Sampling)

Mỗi phần tử có **xác suất như nhau** được chọn.

```python
# Tổng thể: 1000 sinh viên (mã số 1..1000)
population_ids = np.arange(1, 1001)

# Lấy mẫu 50 sinh viên (không hoàn lại)
rng = np.random.default_rng(42)
sample_ids = rng.choice(population_ids, size=50, replace=False)
print("Mẫu 50 sinh viên:", sample_ids[:6], "...")
```

**Ưu điểm:** Đơn giản, công bằng
**Nhược điểm:** Cần danh sách đầy đủ tổng thể

### 13.2.2 Lấy mẫu phân tầng (Stratified Sampling)

Chia tổng thể thành **các tầng** (strata), sau đó lấy mẫu từ mỗi tầng.

**Ví dụ:** Khảo sát sinh viên, chia theo khoa

```python
# Tổng thể: 1000 sinh viên từ 4 khoa
rng = np.random.default_rng(42)
students = pd.DataFrame({
    "id": np.arange(1, 1001),
    "khoa": rng.choice(["CNTT", "KT", "NN", "SP"], size=1000),
})

# Số lượng mỗi khoa
print(students["khoa"].value_counts().sort_index())

# Lấy 10% từ mỗi khoa (R: dplyr group_by + sample_frac)
stratified_sample = students.groupby("khoa").sample(frac=0.1, random_state=42)

print("Mẫu phân tầng:")
print(stratified_sample["khoa"].value_counts().sort_index())
```

**Ưu điểm:** Đảm bảo đại diện cho các nhóm
**Nhược điểm:** Cần biết cấu trúc tổng thể

### 13.2.3 Lấy mẫu cụm (Cluster Sampling)

Chia tổng thể thành **các cụm**, chọn ngẫu nhiên một số cụm.

**Ví dụ:** Khảo sát sinh viên, chọn ngẫu nhiên 5 lớp

```python
# Tổng thể: 20 lớp, mỗi lớp 50 sinh viên
rng = np.random.default_rng(42)

# Chọn ngẫu nhiên 5 lớp
selected_classes = rng.choice(np.arange(1, 21), size=5, replace=False)
print("Các lớp được chọn:", np.sort(selected_classes))

# Khảo sát tất cả sinh viên trong 5 lớp đó
# Tổng: 5 × 50 = 250 sinh viên
```

**Ưu điểm:** Tiết kiệm chi phí, dễ thực hiện
**Nhược điểm:** Sai số lớn hơn nếu các cụm không đồng nhất

### 13.2.4 Lấy mẫu hệ thống (Systematic Sampling)

Chọn phần tử đầu tiên ngẫu nhiên, sau đó chọn **mỗi k phần tử**.

```python
# Tổng thể: 1000 sinh viên
# Muốn lấy mẫu 100 → k = 1000/100 = 10

rng = np.random.default_rng(42)
start = rng.integers(1, 11)                      # Chọn điểm bắt đầu ngẫu nhiên trong 1..10
systematic_sample = np.arange(start, 1001, 10)   # R: seq(start, 1000, by = 10)

print("Bắt đầu từ:", start)
print("Mẫu:", systematic_sample[:6], "...")
print("Kích thước mẫu:", len(systematic_sample))
```

**Ưu điểm:** Dễ thực hiện
**Nhược điểm:** Có thể bị sai lệch nếu có chu kỳ trong dữ liệu

---

## 13.3 Phân phối mẫu và Sai số chuẩn

### 13.3.1 Phân phối mẫu (Sampling Distribution)

**Phân phối mẫu** là phân phối của một **thống kê** (như x̄) từ tất cả các mẫu có thể.

**Ví dụ:** Lấy 1000 mẫu, mỗi mẫu 30 phần tử, tính x̄ của mỗi mẫu.

```python
rng = np.random.default_rng(42)

# Tổng thể
population = rng.normal(100, 15, 100_000)

# Lấy 1000 mẫu, mỗi mẫu 30 phần tử (R: replicate)
n_samples = 1000
sample_size = 30

sample_means = np.array([
    rng.choice(population, sample_size, replace=False).mean()
    for _ in range(n_samples)
])

# Vẽ phân phối mẫu
plt.hist(sample_means, bins=30, density=True, color="lightblue", edgecolor="white")

# Thêm đường cong lý thuyết N(100, 15/√30)
xs = np.linspace(sample_means.min(), sample_means.max(), 200)
plt.plot(xs, stats.norm.pdf(xs, 100, 15 / np.sqrt(30)), color="red", linewidth=2)

plt.title("Phân phối mẫu của x̄")
plt.xlabel("Trung bình mẫu")
plt.show()
```

### 13.3.2 Sai số chuẩn (Standard Error)

**Sai số chuẩn** (SE) là độ lệch chuẩn của phân phối mẫu.

**Công thức:**
```
SE = σ / √n
```

Trong đó:
- σ: Độ lệch chuẩn tổng thể
- n: Kích thước mẫu

**Nếu không biết σ, dùng s (SD mẫu):**
```
SE = s / √n
```

```python
# Ví dụ
sample_data = np.array([85, 90, 78, 92, 88, 95, 80, 89, 91, 87])

s = sample_data.std(ddof=1)
n = len(sample_data)
se = s / np.sqrt(n)

print("SD mẫu (s):", round(s, 2))
print("Kích thước mẫu (n):", n)
print("Sai số chuẩn (SE):", round(se, 2))

# scipy có sẵn hàm tính SE
print("SE (stats.sem):", round(stats.sem(sample_data), 2))
```

**Ý nghĩa:**
- SE **nhỏ**: x̄ gần μ
- SE **lớn**: x̄ khác xa μ
- SE **giảm** khi n **tăng**

---

## 13.4 Khoảng tin cậy (Confidence Interval)

### 13.4.1 Định nghĩa

**Khoảng tin cậy** là một khoảng giá trị có **xác suất chứa tham số** tổng thể.

**Khoảng tin cậy 95% cho μ:**
```
x̄ ± 1.96 × SE
```

**Giải thích:** 95% các khoảng như vậy sẽ chứa μ.

### 13.4.2 Khoảng tin cậy cho trung bình (σ biết)

**Công thức:**
```
CI = x̄ ± z(α/2) × (σ / √n)
```

Với:
- z(α/2): Giá trị z tương ứng mức tin cậy
- 90%: z = 1.645
- 95%: z = 1.96
- 99%: z = 2.576

**Ví dụ:**

```python
# Mẫu 100 sinh viên, x̄ = 75, σ = 10 (biết trước)
xbar = 75
sigma = 10
n = 100

# Khoảng tin cậy 95%
z = stats.norm.ppf(0.975)  # 1.96  (R: qnorm(0.975))
se = sigma / np.sqrt(n)
margin = z * se

lower = xbar - margin
upper = xbar + margin

print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
```

### 13.4.3 Khoảng tin cậy cho trung bình (σ không biết)

Dùng **phân phối t** thay vì z.

**Công thức:**
```
CI = x̄ ± t(α/2, df) × (s / √n)
```

Với df = n - 1 (bậc tự do)

```python
# Ví dụ
sample_data = np.array([85, 90, 78, 92, 88, 95, 80, 89, 91, 87])

# Cách 1: Tự tính
xbar = sample_data.mean()
s = sample_data.std(ddof=1)
n = len(sample_data)
df = n - 1

# t-value cho 95% CI (R: qt(0.975, df))
t_value = stats.t.ppf(0.975, df)
se = s / np.sqrt(n)
margin = t_value * se

lower = xbar - margin
upper = xbar + margin
print(f"95% CI (tự tính):     [{lower:.2f}, {upper:.2f}]")

# Cách 2: Dùng stats.t.interval
ci = stats.t.interval(0.95, df, loc=xbar, scale=stats.sem(sample_data))
print(f"95% CI (t.interval):  [{ci[0]:.2f}, {ci[1]:.2f}]")

# Cách 3: Lấy từ kết quả kiểm định t (R: t.test(x)$conf.int)
ci = stats.ttest_1samp(sample_data, popmean=0).confidence_interval(0.95)
print(f"95% CI (ttest_1samp): [{ci.low:.2f}, {ci.high:.2f}]")
```

### 13.4.4 Giải thích khoảng tin cậy

**SAI:** "Xác suất μ nằm trong [80, 90] là 95%"
**ĐÚNG:** "95% các khoảng tin cậy tính theo cách này sẽ chứa μ"

**Minh họa:**

```python
rng = np.random.default_rng(42)

# Tổng thể với μ = 100
population = rng.normal(100, 15, 100_000)

# Tính 100 khoảng tin cậy
n_intervals = 100
sample_size = 30

rows = []
for i in range(n_intervals):
    sample_data = rng.choice(population, sample_size, replace=False)
    ci = stats.t.interval(0.95, sample_size - 1,
                          loc=sample_data.mean(), scale=stats.sem(sample_data))
    rows.append({"lower": ci[0], "upper": ci[1],
                 "contains_mu": ci[0] <= 100 <= ci[1]})

results = pd.DataFrame(rows)

# Đếm bao nhiêu khoảng chứa μ
count = results["contains_mu"].sum()
print(f"Số khoảng chứa μ = 100: {count} / {n_intervals}")
print(f"Tỷ lệ: {count / n_intervals * 100:.0f}%")

# Trực quan hóa: khoảng màu đỏ là khoảng KHÔNG chứa μ
colors = np.where(results["contains_mu"], "steelblue", "red")
plt.hlines(y=results.index, xmin=results["lower"], xmax=results["upper"], colors=colors)
plt.axvline(100, color="black", linestyle="--")
plt.title("100 khoảng tin cậy 95% (đỏ = không chứa μ)")
plt.xlabel("Giá trị")
plt.ylabel("Mẫu thứ")
plt.show()
```

---

## 13.5 Kiểm định giả thuyết (Hypothesis Testing)

### 13.5.1 Khái niệm cơ bản

**Giả thuyết không (H₀):** Giả thuyết ban đầu, thường là "không có sự khác biệt"

**Giả thuyết đối (H₁ hoặc Hₐ):** Giả thuyết mà ta muốn chứng minh

**Ví dụ:**
- H₀: μ = 100 (IQ trung bình = 100)
- H₁: μ ≠ 100 (IQ trung bình khác 100)

### 13.5.2 Các loại kiểm định

**1. Kiểm định hai đuôi (Two-tailed)**
- H₁: μ ≠ μ₀ → trong scipy: `alternative="two-sided"` (mặc định)

**2. Kiểm định đuôi phải (Right-tailed)**
- H₁: μ > μ₀ → `alternative="greater"`

**3. Kiểm định đuôi trái (Left-tailed)**
- H₁: μ < μ₀ → `alternative="less"`

```python
# Minh họa
x = np.linspace(-4, 4, 400)
y = stats.norm.pdf(x)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
settings = [
    ("Hai đuôi",  lambda v: (v <= -1.96) | (v >= 1.96), [-1.96, 1.96]),
    ("Đuôi phải", lambda v: v >= 1.645, [1.645]),
    ("Đuôi trái", lambda v: v <= -1.645, [-1.645]),
]
for ax, (title, region, cuts) in zip(axes, settings):
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, where=region(x), color="red", alpha=0.4)   # Vùng bác bỏ
    for c in cuts:
        ax.axvline(c, color="red", linestyle="--")
    ax.set_title(title)
    ax.set_yticks([])

plt.tight_layout()
plt.show()
```

### 13.5.3 P-value

**P-value** là xác suất quan sát được dữ liệu **cực đoan như vậy hoặc hơn**, nếu H₀ đúng.

**Quy tắc:**
- p-value < α: **Bác bỏ H₀**
- p-value ≥ α: **Không bác bỏ H₀**

Thường dùng α = 0.05 (mức ý nghĩa 5%)

### 13.5.4 Các bước kiểm định

**Bước 1:** Thiết lập giả thuyết H₀ và H₁

**Bước 2:** Chọn mức ý nghĩa α (thường 0.05)

**Bước 3:** Tính thống kê kiểm định (t, z, chi-square, ...)

**Bước 4:** Tính p-value

**Bước 5:** Kết luận
- p < α: Bác bỏ H₀
- p ≥ α: Không bác bỏ H₀

### 13.5.5 Loại sai lầm

**Sai lầm loại I (Type I Error):**
- Bác bỏ H₀ khi H₀ **đúng**
- Xác suất = α

**Sai lầm loại II (Type II Error):**
- Không bác bỏ H₀ khi H₀ **sai**
- Xác suất = β

|  | H₀ đúng | H₀ sai |
|--|---------|--------|
| **Bác bỏ H₀** | Sai lầm loại I (α) | Đúng (Power = 1-β) |
| **Không bác bỏ H₀** | Đúng | Sai lầm loại II (β) |

---

## 13.6 Kiểm định t (t-test)

> **Hàm tiện ích in kết luận** – dùng lại cho các ví dụ bên dưới:
> ```python
> def ket_luan(p_value, bac_bo, khong_bac_bo, alpha=0.05):
>     if p_value < alpha:
>         print(f"p-value = {p_value:.4f} < {alpha}")
>         print("Bác bỏ H0:", bac_bo)
>     else:
>         print(f"p-value = {p_value:.4f} ≥ {alpha}")
>         print("Không bác bỏ H0:", khong_bac_bo)
> ```

```python
def ket_luan(p_value, bac_bo, khong_bac_bo, alpha=0.05):
    if p_value < alpha:
        print(f"p-value = {p_value:.4f} < {alpha}")
        print("Bác bỏ H0:", bac_bo)
    else:
        print(f"p-value = {p_value:.4f} ≥ {alpha}")
        print("Không bác bỏ H0:", khong_bac_bo)
```

### 13.6.1 Kiểm định t một mẫu (One-sample t-test)

**Mục đích:** Kiểm tra xem trung bình mẫu có **khác** với một giá trị cho trước không.

**Giả thuyết:**
- H₀: μ = μ₀
- H₁: μ ≠ μ₀

**Thống kê:**
```
t = (x̄ - μ₀) / (s / √n)
```

**Ví dụ:** Chiều cao nam sinh viên có trung bình = 170cm?

```python
# Dữ liệu: Chiều cao 20 nam sinh viên
heights = np.array([168, 172, 165, 175, 170, 173, 169, 171, 174, 167,
                    172, 170, 168, 176, 171, 169, 173, 170, 174, 172])

# H0: μ = 170
# H1: μ ≠ 170

# Kiểm định (R: t.test(heights, mu = 170))
result = stats.ttest_1samp(heights, popmean=170)
ci = result.confidence_interval(0.95)
print(f"t = {result.statistic:.4f}, df = {result.df}, p-value = {result.pvalue:.4f}")
print(f"95% CI: [{ci.low:.2f}, {ci.high:.2f}], mean = {heights.mean():.2f}")

print("\nKết luận:")
ket_luan(result.pvalue,
         "Chiều cao TB KHÁC 170cm",
         "Chiều cao TB có thể = 170cm")
```

### 13.6.2 Kiểm định t hai mẫu độc lập (Independent two-sample t-test)

**Mục đích:** So sánh trung bình của **hai nhóm độc lập**.

**Giả thuyết:**
- H₀: μ₁ = μ₂
- H₁: μ₁ ≠ μ₂

**Ví dụ:** So sánh điểm thi giữa nam và nữ

```python
# Dữ liệu
male_scores = np.array([75, 80, 72, 85, 78, 82, 76, 79, 81, 77])
female_scores = np.array([82, 85, 80, 88, 83, 86, 84, 87, 85, 83])

# H0: μ_nam = μ_nữ
# H1: μ_nam ≠ μ_nữ

# Kiểm định Welch (không giả định phương sai bằng nhau) – giống MẶC ĐỊNH của t.test() trong R
# Chú ý: scipy mặc định equal_var=True, nên phải ghi rõ equal_var=False
result = stats.ttest_ind(male_scores, female_scores, equal_var=False)
print(f"t = {result.statistic:.4f}, df = {result.df:.2f}, p-value = {result.pvalue:.6f}")

print("\nThống kê mô tả:")
print(f"Nam - Mean: {male_scores.mean()}  SD: {male_scores.std(ddof=1):.2f}")
print(f"Nữ  - Mean: {female_scores.mean()}  SD: {female_scores.std(ddof=1):.2f}")

print("\nKết luận:")
ket_luan(result.pvalue,
         "Có sự KHÁC BIỆT giữa nam và nữ",
         "KHÔNG có sự khác biệt")
```

### 13.6.3 Kiểm định t hai mẫu ghép đôi (Paired t-test)

**Mục đích:** So sánh **trước và sau** trong cùng nhóm.

**Giả thuyết:**
- H₀: μ_diff = 0
- H₁: μ_diff ≠ 0

**Ví dụ:** Điểm thi trước và sau khóa học

```python
# Dữ liệu
before = np.array([65, 70, 68, 72, 69, 71, 67, 73, 70, 68])
after = np.array([70, 75, 72, 78, 74, 76, 71, 77, 75, 73])

# H0: μ_after - μ_before = 0
# H1: μ_after - μ_before ≠ 0

# Kiểm định (R: t.test(after, before, paired = TRUE))
result = stats.ttest_rel(after, before)
print(f"t = {result.statistic:.4f}, df = {result.df}, p-value = {result.pvalue:.2e}")

# Tính chênh lệch
diff = after - before
print("\nChênh lệch trung bình:", diff.mean())
print("SD chênh lệch:", round(diff.std(ddof=1), 4))

print("\nKết luận:")
ket_luan(result.pvalue,
         "Điểm SAU cao hơn TRƯỚC",
         "KHÔNG có sự cải thiện")
```

---

## 13.7 Kiểm định Chi-square

### 13.7.1 Kiểm định Chi-square độc lập

**Mục đích:** Kiểm tra **mối quan hệ** giữa hai biến phân loại.

**Giả thuyết:**
- H₀: Hai biến **độc lập**
- H₁: Hai biến **có quan hệ**

**Ví dụ:** Giới tính và sở thích môn học

```python
# Tạo bảng chéo
data = pd.DataFrame([[30, 10, 15],
                     [25, 20, 30]],
                    index=["Nam", "Nữ"],
                    columns=["Toán", "Văn", "Anh"])

print("Bảng chéo:")
print(data)

# H0: Giới tính và sở thích độc lập
# H1: Giới tính và sở thích có quan hệ

# Kiểm định (R: chisq.test(data))
result = stats.chi2_contingency(data)
print(f"\nX-squared = {result.statistic:.4f}, df = {result.dof}, p-value = {result.pvalue:.4f}")

print("\nKết luận:")
ket_luan(result.pvalue,
         "Giới tính VÀ sở thích CÓ QUAN HỆ",
         "Giới tính và sở thích ĐỘC LẬP")

# Xem tần số kỳ vọng
print("\nTần số kỳ vọng:")
print(pd.DataFrame(result.expected_freq, index=data.index, columns=data.columns).round(2))
```

> **Lưu ý:** Với bảng 2×2, cả R và `chi2_contingency` đều mặc định áp dụng hiệu chỉnh liên tục Yates (`correction=True`).

### 13.7.2 Kiểm định Chi-square phù hợp (Goodness-of-fit)

**Mục đích:** Kiểm tra xem dữ liệu có **phù hợp** với phân phối lý thuyết không.

**Ví dụ:** Xúc xắc có cân đối không?

```python
# Tung xúc xắc 120 lần
observed = pd.Series([18, 22, 19, 21, 20, 20], index=[1, 2, 3, 4, 5, 6])

print("Tần số quan sát:")
print(observed.to_dict())

# H0: Xúc xắc cân đối (mỗi mặt p = 1/6)
# H1: Xúc xắc không cân đối

# Tần số kỳ vọng (nếu cân đối)
expected = np.repeat(120 / 6, 6)
print("\nTần số kỳ vọng:", expected)

# Kiểm định (R: chisq.test(observed, p = rep(1/6, 6)))
result = stats.chisquare(f_obs=observed, f_exp=expected)
print(f"\nX-squared = {result.statistic:.2f}, df = 5, p-value = {result.pvalue:.4f}")

print("\nKết luận:")
ket_luan(result.pvalue,
         "Xúc xắc KHÔNG cân đối",
         "Xúc xắc CÂN ĐỐI")
```

---

## 13.8 Ví dụ tổng hợp

### Ví dụ 1: Nghiên cứu thuốc mới

**Bối cảnh:** Thử nghiệm thuốc giảm cholesterol trên 30 bệnh nhân.

```python
rng = np.random.default_rng(123)

# Cholesterol trước và sau dùng thuốc
before = rng.normal(220, 20, 30)
after = before - rng.normal(15, 8, 30)

print("=== NGHIÊN CỨU THUỐC MỚI ===\n")

# 1. Thống kê mô tả
print("1. THỐNG KÊ MÔ TẢ")
print(f"Trước - Mean: {before.mean():.2f}  SD: {before.std(ddof=1):.2f}")
print(f"Sau   - Mean: {after.mean():.2f}  SD: {after.std(ddof=1):.2f}")
print(f"Chênh lệch TB: {before.mean() - after.mean():.2f}\n")

# 2. Kiểm định paired t-test
print("2. KIỂM ĐỊNH")
print("H0: Thuốc KHÔNG có tác dụng (μ_diff = 0)")
print("H1: Thuốc CÓ tác dụng (μ_diff ≠ 0)\n")

result = stats.ttest_rel(before, after)
ci = result.confidence_interval(0.95)
print(f"t = {result.statistic:.3f}, df = {result.df}, p-value = {result.pvalue:.2e}")

# 3. Kết luận
print("\n3. KẾT LUẬN")
if result.pvalue < 0.05:
    print("p-value < 0.05 → BÁC BỎ H0")
    print("Thuốc CÓ HIỆU QUẢ giảm cholesterol")
    print(f"Giảm trung bình: {(before - after).mean():.2f} mg/dL")
    print(f"95% CI: [{ci.low:.2f}, {ci.high:.2f}]")
else:
    print("p-value ≥ 0.05 → KHÔNG BÁC BỎ H0")
    print("CHƯA có bằng chứng thuốc có hiệu quả")

# 4. Vẽ biểu đồ
bp = plt.boxplot([before, after], tick_labels=["Trước", "Sau"], patch_artist=True)
for box, c in zip(bp["boxes"], ["lightcoral", "lightblue"]):
    box.set_facecolor(c)
plt.title("Cholesterol trước và sau dùng thuốc")
plt.ylabel("Cholesterol (mg/dL)")
plt.show()
```

### Ví dụ 2: Khảo sát A/B Testing

**Bối cảnh:** So sánh tỷ lệ click giữa 2 phiên bản website.

```python
from statsmodels.stats.proportion import proportions_ztest

# Dữ liệu
# Phiên bản A: 1000 người, 150 click
# Phiên bản B: 1000 người, 180 click

print("=== A/B TESTING ===\n")

n_A, clicks_A = 1000, 150
n_B, clicks_B = 1000, 180

# Tỷ lệ
p_A = clicks_A / n_A
p_B = clicks_B / n_B

print(f"Phiên bản A: Tỷ lệ click = {p_A * 100:.1f}%")
print(f"Phiên bản B: Tỷ lệ click = {p_B * 100:.1f}%\n")

# Kiểm định tỷ lệ
# H0: p_A = p_B
# H1: p_A ≠ p_B

# Cách 1: z-test cho 2 tỷ lệ (statsmodels)
z_stat, p_value = proportions_ztest(count=[clicks_A, clicks_B], nobs=[n_A, n_B])
print(f"z-test:   z = {z_stat:.3f}, p-value = {p_value:.4f}")

# Cách 2: Chi-square trên bảng 2x2 có hiệu chỉnh Yates → KẾT QUẢ GIỐNG prop.test() của R
table = [[clicks_A, n_A - clicks_A],
         [clicks_B, n_B - clicks_B]]
chi = stats.chi2_contingency(table, correction=True)
print(f"prop.test: X-squared = {chi.statistic:.3f}, p-value = {chi.pvalue:.4f}")

print("\nKẾT LUẬN (theo prop.test):")
if chi.pvalue < 0.05:
    print("p-value < 0.05 → CÓ SỰ KHÁC BIỆT")
    print("Phiên bản B TỐT HƠN phiên bản A" if p_B > p_A else "Phiên bản A TỐT HƠN phiên bản B")
else:
    print("p-value ≥ 0.05 → KHÔNG có sự khác biệt có ý nghĩa thống kê")
    print("Chưa đủ bằng chứng để kết luận phiên bản nào tốt hơn")
```

> **Nhận xét:** Hai cách cho p-value hơi khác nhau vì cách 2 có hiệu chỉnh liên tục (thận trọng hơn). Với p-value quanh ngưỡng 0.05, kết luận có thể thay đổi tùy phương pháp – đây là lý do cần báo cáo rõ phương pháp kiểm định đã dùng.

---

## BÀI TẬP THỰC HÀNH

### Bài tập 1: Khoảng tin cậy

Một mẫu 25 sinh viên có điểm TB = 78, SD = 8.

1. Tính 95% CI cho điểm TB tổng thể (gợi ý: `stats.t.interval(0.95, df=24, loc=78, scale=8/np.sqrt(25))`)
2. Tính 99% CI
3. Giải thích ý nghĩa của CI

### Bài tập 2: One-sample t-test

Nhà máy sản xuất bóng đèn tuyên bố tuổi thọ TB = 1000 giờ. Kiểm tra 20 bóng đèn:
```
980, 1020, 990, 1010, 1005, 995, 1015, 985, 1000, 1010,
995, 1005, 1000, 990, 1015, 1010, 995, 1000, 1005, 1010
```

Kiểm định với α = 0.05: Tuổi thọ TB có = 1000 giờ?

### Bài tập 3: Independent t-test

So sánh điểm thi giữa 2 lớp:
- Lớp A: 75, 80, 72, 85, 78, 82, 76, 79, 81, 77
- Lớp B: 82, 85, 80, 88, 83, 86, 84, 87, 85, 83

Có sự khác biệt không? (Nhớ `equal_var=False` để dùng kiểm định Welch)

### Bài tập 4: Paired t-test

Cân nặng trước và sau chế độ ăn kiêng (10 người):
- Trước: 70, 75, 68, 80, 72, 77, 69, 74, 76, 71
- Sau: 68, 72, 66, 77, 70, 74, 67, 71, 73, 69

Chế độ ăn có hiệu quả không? (Gợi ý: thử `alternative="greater"` cho kiểm định một phía)

### Bài tập 5: Chi-square

Khảo sát 200 người về sở thích phim:

|  | Hành động | Tâm lý | Hài |
|--|-----------|--------|-----|
| **Nam** | 40 | 30 | 30 |
| **Nữ** | 20 | 50 | 30 |

Giới tính và sở thích phim có quan hệ không?

---

## CÂU HỎI ÔN TẬP

1. Phân biệt tổng thể và mẫu?
2. Sai số chuẩn (SE) là gì? Khác gì với SD?
3. Giải thích ý nghĩa của khoảng tin cậy 95%?
4. Phân biệt H₀ và H₁?
5. P-value là gì? Khi nào bác bỏ H₀?
6. Phân biệt sai lầm loại I và loại II?
7. Khi nào dùng z-test? Khi nào dùng t-test?
8. Paired t-test khác gì independent t-test?

---

## TÀI LIỆU THAM KHẢO

1. **SciPy Documentation**: [`ttest_1samp`, `ttest_ind`, `ttest_rel`, `chi2_contingency`, `chisquare`](https://docs.scipy.org/doc/scipy/reference/stats.html#hypothesis-tests-and-related-functions)
2. **statsmodels**: [`proportions_ztest`](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)
3. **Statistics by Jim**: https://statisticsbyjim.com/
4. **Khan Academy**: Hypothesis Testing

---

## TỔNG KẾT

### Công thức tổng hợp:

**Sai số chuẩn:**
```
SE = s / √n
```

**Khoảng tin cậy:**
```
CI = x̄ ± t(α/2, df) × SE
```

**Thống kê t:**
```
t = (x̄ - μ₀) / SE
```

### Quy trình kiểm định:

1. **Thiết lập**: H₀ và H₁
2. **Chọn**: Mức ý nghĩa α
3. **Tính**: Thống kê kiểm định
4. **Tính**: p-value
5. **Kết luận**: So sánh p với α

### Các kiểm định chính:

| Kiểm định | Mục đích | Python | R |
|-----------|----------|-------|---|
| One-sample t | μ = μ₀? | `stats.ttest_1samp(x, popmean)` | `t.test(x, mu)` |
| Two-sample t (Welch) | μ₁ = μ₂? | `stats.ttest_ind(x, y, equal_var=False)` | `t.test(x, y)` |
| Paired t | Trước = Sau? | `stats.ttest_rel(x, y)` | `t.test(x, y, paired=TRUE)` |
| Chi-square độc lập | Độc lập? | `stats.chi2_contingency(table)` | `chisq.test(table)` |
| Chi-square phù hợp | Phù hợp phân phối? | `stats.chisquare(obs, exp)` | `chisq.test(x, p)` |
| Proportion | p₁ = p₂? | `proportions_ztest()` (statsmodels) | `prop.test()` |
| Normality | Chuẩn? | `stats.shapiro(x)` | `shapiro.test(x)` |

### Lưu ý:

✅ **α = 0.05**: Mức ý nghĩa thường dùng
✅ **p < 0.05**: Bác bỏ H₀
✅ **CI 95%**: Khoảng tin cậy phổ biến
✅ **n ≥ 30**: Mẫu lớn
✅ **Paired**: Cùng đối tượng, trước/sau
✅ **Independent**: Hai nhóm khác nhau – **nhớ `equal_var=False`** để giống R

---

**Cập nhật**: Tháng 3/2026
