# BÀI 9: TRỰC QUAN HÓA DỮ LIỆU CƠ BẢN VỚI MATPLOTLIB

## Mục tiêu học tập
- Hiểu tầm quan trọng của việc trực quan hóa dữ liệu
- Nắm vững các loại biểu đồ cơ bản trong Python (thư viện `matplotlib`)
- Biết cách tùy chỉnh biểu đồ (màu sắc, nhãn, tiêu đề)
- Vẽ được nhiều biểu đồ cùng lúc để so sánh
- Áp dụng biểu đồ phù hợp cho từng loại dữ liệu

---

## 9.0 Chuẩn bị

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm   # Chỉ dùng để tải các bộ dữ liệu mẫu của R

mtcars = sm.datasets.get_rdataset("mtcars").data
iris = sm.datasets.get_rdataset("iris").data
anscombe = sm.datasets.get_rdataset("anscombe").data
```

**Hai cách vẽ với matplotlib:**

| Cách | Ví dụ | Khi nào dùng |
| :--- | :--- | :--- |
| Kiểu "pyplot" (giống R base graphics) | `plt.hist(x)`, `plt.title("...")` | Biểu đồ đơn, nhanh |
| Kiểu hướng đối tượng (khuyên dùng) | `fig, ax = plt.subplots()` rồi `ax.hist(x)`, `ax.set_title("...")` | Nhiều biểu đồ, tùy chỉnh chi tiết |

> Trong Jupyter, biểu đồ hiển thị ngay dưới ô code. Khi chạy file `.py`, cần gọi `plt.show()` ở cuối để hiện cửa sổ biểu đồ.

---

## 9.1 Tại sao cần trực quan hóa dữ liệu?

### Trực quan hóa giúp:
- **Hiểu dữ liệu nhanh chóng**: Một biểu đồ có thể cho ta thấy toàn cảnh ngay lập tức
- **Phát hiện xu hướng, mẫu hình** (patterns): Dễ nhận ra xu hướng tăng/giảm
- **Tìm ra giá trị bất thường** (outliers): Phát hiện dữ liệu không bình thường
- **Truyền đạt thông tin hiệu quả**: "Một hình ảnh đáng giá ngàn lời"
- **Ra quyết định tốt hơn**: Dựa trên bằng chứng trực quan

### Ví dụ minh họa: Anscombe's Quartet

4 tập dữ liệu có cùng thống kê mô tả nhưng rất khác nhau khi vẽ

```python
# Xem dữ liệu
print(anscombe)

# Tính thống kê cho x1 và y1
print("Trung bình x1:", anscombe["x1"].mean())
print("Trung bình y1:", round(anscombe["y1"].mean(), 3))
print("Độ lệch chuẩn x1:", round(anscombe["x1"].std(), 3))
print("Độ lệch chuẩn y1:", round(anscombe["y1"].std(), 3))
print("Tương quan:", round(anscombe["x1"].corr(anscombe["y1"]), 3))

# Thống kê tương tự cho các cặp khác
# Nhưng khi vẽ ra sẽ thấy rất khác nhau!
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for i, ax in enumerate(axes.flat, start=1):
    ax.scatter(anscombe[f"x{i}"], anscombe[f"y{i}"], color="steelblue")
    ax.set_title(f"Tập {i}")
plt.tight_layout()
plt.show()
```

---

## 9.2 Biểu đồ cột (Bar Plot)

### 9.2.1 Biểu đồ cột cơ bản

**Biểu đồ cột** được sử dụng để hiển thị **dữ liệu phân loại** (categorical data)

**Ví dụ:** Đếm số lượng xe theo số xy-lanh

```python
# Đếm số xe theo số xy-lanh (R: table(mtcars$cyl))
table_cyl = mtcars["cyl"].value_counts().sort_index()
print(table_cyl)
# cyl
# 4    11
# 6     7
# 8    14

# Vẽ biểu đồ cột (trục x dùng chuỗi để các cột cách đều nhau)
plt.bar(table_cyl.index.astype(str), table_cyl.values)
plt.show()
```

### 9.2.2 Tùy chỉnh biểu đồ cột

```python
# Thêm tiêu đề và nhãn trục
plt.bar(table_cyl.index.astype(str), table_cyl.values, color="steelblue")  # Màu cột
plt.title("Số lượng xe theo số xy-lanh")   # Tiêu đề
plt.xlabel("Số xy-lanh")                   # Nhãn trục x
plt.ylabel("Số lượng xe")                  # Nhãn trục y
plt.show()

# Thay đổi màu sắc cho mỗi cột
plt.bar(table_cyl.index.astype(str), table_cyl.values, color=["red", "green", "blue"])
plt.title("Số lượng xe theo số xy-lanh")
plt.xlabel("Số xy-lanh")
plt.ylabel("Số lượng xe")
plt.show()

# Biểu đồ cột ngang (horizontal) – dùng plt.barh
plt.barh(table_cyl.index.astype(str), table_cyl.values, color="orange")
plt.title("Số lượng xe theo số xy-lanh")
plt.xlabel("Số lượng xe")
plt.ylabel("Số xy-lanh")
plt.show()
```

### 9.2.3 Biểu đồ cột cho dữ liệu số

```python
# Ví dụ: Điểm trung bình của 5 sinh viên
students = ["Tùng", "Hùng", "Dũng", "Linh", "Mai"]
scores = [8.5, 9.0, 7.5, 8.8, 9.2]

plt.bar(students, scores,          # Tên cho mỗi cột (R: names.arg)
        color="skyblue",
        edgecolor="darkblue")      # Màu viền (R: border)
plt.title("Điểm trung bình sinh viên")
plt.xlabel("Sinh viên")
plt.ylabel("Điểm")
plt.ylim(0, 10)                    # Giới hạn trục y
plt.show()

# Thêm giá trị lên đầu mỗi cột
fig, ax = plt.subplots()
bars = ax.bar(students, scores, color="lightgreen")
ax.set_title("Điểm trung bình sinh viên")
ax.set_xlabel("Sinh viên")
ax.set_ylabel("Điểm")
ax.set_ylim(0, 10)

# Thêm text: bar_label tự tính vị trí (không cần tự tính tọa độ như text() của R)
ax.bar_label(bars, labels=scores, padding=3)
plt.show()
```

### 9.2.4 Biểu đồ cột nhóm (Grouped Bar Plot)

Với dữ liệu dạng bảng (DataFrame), pandas cung cấp sẵn `df.plot(kind="bar")` giúp vẽ biểu đồ nhóm rất nhanh.

```python
# Ví dụ: So sánh điểm của 2 lớp
scores_df = pd.DataFrame({
    "Lớp A": [8.5, 7.8, 9.0, 8.2, 7.5],
    "Lớp B": [8.0, 8.5, 8.8, 7.9, 8.3],
}, index=["Môn 1", "Môn 2", "Môn 3", "Môn 4", "Môn 5"])

# Cột cạnh nhau (R: beside = TRUE)
scores_df.plot(kind="bar", color=["lightblue", "lightcoral"], rot=0)
plt.title("So sánh điểm 2 lớp")
plt.xlabel("Môn học")
plt.ylabel("Điểm")
plt.show()

# Biểu đồ cột chồng (Stacked Bar Plot)
scores_df.plot(kind="bar", stacked=True, color=["lightblue", "lightcoral"], rot=0)
plt.title("So sánh điểm 2 lớp (Stacked)")
plt.xlabel("Môn học")
plt.ylabel("Tổng điểm")
plt.show()
```

---

## 9.3 Biểu đồ tần số (Histogram)

### 9.3.1 Histogram cơ bản

**Histogram** được sử dụng để hiển thị **phân phối** của dữ liệu liên tục

**Khác với barplot:**
- Histogram dùng cho **dữ liệu số liên tục**
- Barplot dùng cho **dữ liệu phân loại**

```python
# Ví dụ: Phân phối mpg (miles per gallon)
plt.hist(mtcars["mpg"])
plt.show()
```

### 9.3.2 Tùy chỉnh Histogram

```python
# Thêm tiêu đề và nhãn
plt.hist(mtcars["mpg"], color="lightblue", edgecolor="darkblue")
plt.title("Phân phối tiêu thụ nhiên liệu")
plt.xlabel("Miles per Gallon")
plt.ylabel("Tần số")
plt.show()

# Thay đổi số lượng bins (cột)
plt.hist(mtcars["mpg"], bins=5, color="coral", edgecolor="white")   # R: breaks = 5
plt.title("Histogram với 5 bins")
plt.xlabel("MPG")
plt.show()

plt.hist(mtcars["mpg"], bins=15, color="lightgreen", edgecolor="white")
plt.title("Histogram với 15 bins")
plt.xlabel("MPG")
plt.show()
```

> **Lưu ý:** matplotlib mặc định dùng **10 bins** và **không vẽ viền cột** – nên thêm `edgecolor="white"` hoặc `"black"` để dễ nhìn.

### 9.3.3 Histogram với đường cong mật độ

```python
from scipy.stats import gaussian_kde

# Tạo histogram với mật độ xác suất thay vì tần số
plt.hist(mtcars["mpg"], color="lightgray", edgecolor="white",
         density=True)   # Trục y là mật độ xác suất (R: probability = TRUE)

# Thêm đường cong mật độ (R: lines(density(x)))
kde = gaussian_kde(mtcars["mpg"])
x_grid = np.linspace(mtcars["mpg"].min() - 5, mtcars["mpg"].max() + 5, 200)
plt.plot(x_grid, kde(x_grid), color="red", linewidth=2)   # Độ dày đường (R: lwd)

plt.title("Histogram với đường cong mật độ")
plt.xlabel("MPG")
plt.show()
```

### 9.3.4 Ví dụ thực tế

```python
# Phân phối chiều cao sinh viên (giả định)
rng = np.random.default_rng(42)                    # R: set.seed(42)
heights = rng.normal(loc=165, scale=8, size=100)   # 100 SV, TB=165cm, SD=8cm (R: rnorm)

plt.hist(heights, bins=10, color="skyblue", edgecolor="white")
plt.title("Phân phối chiều cao sinh viên")
plt.xlabel("Chiều cao (cm)")
plt.ylabel("Số sinh viên")

# Thêm đường trung bình (R: abline(v = ...))
plt.axvline(heights.mean(), color="red", linewidth=2, linestyle="--",   # Nét đứt
            label=f"TB = {heights.mean():.1f} cm")

# Thêm chú thích
plt.legend(loc="upper right")
plt.show()
```

---

## 9.4 Biểu đồ hộp (Box Plot)

### 9.4.1 Box Plot là gì?

**Box plot** (còn gọi là box-and-whisker plot) hiển thị:

- **Trung vị** (median): Đường giữa hộp
- **Q1** (25%): Cạnh dưới hộp  
- **Q3** (75%): Cạnh trên hộp
- **IQR = Q3 - Q1**: Chiều cao hộp (khoảng tứ phân vị)
- **Whiskers** (râu): Kéo dài đến 1.5×IQR
- **Outliers** (điểm bất thường): Các điểm ngoài whiskers

```python
plt.boxplot(mtcars["mpg"])
plt.show()
```

### 9.4.2 Tùy chỉnh Box Plot

```python
plt.boxplot(mtcars["mpg"], patch_artist=True,   # patch_artist=True để tô màu hộp
            boxprops=dict(facecolor="lightgreen", color="darkgreen"),
            medianprops=dict(color="darkgreen"))
plt.title("Box Plot của MPG")
plt.ylabel("Miles per Gallon")
plt.show()
```

### 9.4.3 So sánh nhiều nhóm

```python
# So sánh mpg theo số xy-lanh (R: boxplot(mpg ~ cyl, data = mtcars))
groups = [g["mpg"].values for _, g in mtcars.groupby("cyl")]
labels = sorted(mtcars["cyl"].unique())

bp = plt.boxplot(groups, tick_labels=labels, patch_artist=True)
for box, color in zip(bp["boxes"], ["red", "green", "blue"]):
    box.set_facecolor(color)
plt.title("MPG theo số xy-lanh")
plt.xlabel("Số xy-lanh")
plt.ylabel("Miles per Gallon")
plt.show()

# Cách nhanh hơn với pandas:
mtcars.boxplot(column="mpg", by="cyl", grid=False)
plt.suptitle("")   # Bỏ tiêu đề phụ pandas tự thêm
plt.title("MPG theo số xy-lanh")
plt.show()
```

> **Phiên bản matplotlib:** Tham số `tick_labels` có từ matplotlib 3.9. Với phiên bản cũ hơn, dùng `labels=labels`.

**Giải thích:**
- Xe 4 xy-lanh có mpg cao nhất
- Xe 8 xy-lanh có mpg thấp nhất
- Có một số outliers

### 9.4.4 Box Plot ngang

```python
bp = plt.boxplot(groups, tick_labels=labels, patch_artist=True,
                 orientation="horizontal")   # matplotlib < 3.10: dùng vert=False
for box, color in zip(bp["boxes"], ["lightblue", "lightcoral", "lightyellow"]):
    box.set_facecolor(color)
plt.title("MPG theo số xy-lanh (ngang)")
plt.xlabel("Miles per Gallon")
plt.ylabel("Số xy-lanh")
plt.show()
```

### 9.4.5 Ví dụ với dữ liệu iris

```python
# So sánh chiều dài đài hoa giữa các loài
species = iris["Species"].unique()
groups = [iris.loc[iris["Species"] == s, "Sepal.Length"] for s in species]

bp = plt.boxplot(groups, tick_labels=species, patch_artist=True)
for box, color in zip(bp["boxes"], ["pink", "lightblue", "lightgreen"]):
    box.set_facecolor(color)
plt.title("Chiều dài đài hoa theo loài")
plt.xlabel("Loài")
plt.ylabel("Chiều dài đài hoa (cm)")
plt.show()
```

**Nhận xét từ biểu đồ:**
- Virginica có đài hoa dài nhất
- Setosa có đài hoa ngắn nhất
- Có outlier ở Virginica (một cây có đài hoa ngắn bất thường)

### 9.4.6 Nhiều Box Plot cùng lúc

```python
# So sánh tất cả 4 biến số của iris
bp = plt.boxplot(iris.iloc[:, 0:4], patch_artist=True,
                 tick_labels=["Sepal.L", "Sepal.W", "Petal.L", "Petal.W"])
for box, color in zip(bp["boxes"], plt.cm.rainbow(np.linspace(0, 1, 4))):  # R: rainbow(4)
    box.set_facecolor(color)
plt.title("So sánh các đặc điểm của hoa")
plt.show()
```

---

## 9.5 Biểu đồ phân tán (Scatter Plot)

### 9.5.1 Scatter Plot cơ bản

**Scatter plot** hiển thị **mối quan hệ giữa 2 biến số**

```python
# Ví dụ: Mối quan hệ giữa trọng lượng và mpg
plt.scatter(mtcars["wt"], mtcars["mpg"])
plt.show()
```

### 9.5.2 Tùy chỉnh Scatter Plot

```python
plt.scatter(mtcars["wt"], mtcars["mpg"],
            color="blue",
            marker="o")   # Kiểu điểm (R: pch = 19)
plt.title("Mối quan hệ giữa trọng lượng và MPG")
plt.xlabel("Trọng lượng (1000 lbs)")
plt.ylabel("Miles per Gallon")
plt.show()
```

### 9.5.3 Các kiểu điểm khác nhau

Trong R dùng số `pch` (0–25); trong matplotlib dùng tham số `marker` với ký hiệu dạng chuỗi.

```python
markers = ["o", "s", "^", "v", "<", ">", "D", "d", "p", "h",
           "*", "+", "x", "P", "X", "1", "2", "3", "4", "|", "_", "."]

fig, ax = plt.subplots(figsize=(10, 3))
for i, m in enumerate(markers):
    ax.scatter(i, 0, marker=m, s=150, color=plt.cm.tab20(i % 20))  # s: kích thước (R: cex)
    ax.text(i, 0.3, repr(m), ha="center")                          # Thêm nhãn
ax.set_ylim(-0.5, 0.6)
ax.set_yticks([])
ax.set_title("Các kiểu điểm (marker)")
plt.show()
```

### 9.5.4 Thêm đường xu hướng (trend line)

```python
plt.scatter(mtcars["wt"], mtcars["mpg"], color="darkblue")
plt.title("MPG vs Trọng lượng")
plt.xlabel("Trọng lượng (1000 lbs)")
plt.ylabel("Miles per Gallon")

# Thêm đường hồi quy tuyến tính (R: lm + abline)
slope, intercept = np.polyfit(mtcars["wt"], mtcars["mpg"], deg=1)
x_line = np.linspace(mtcars["wt"].min(), mtcars["wt"].max(), 100)
plt.plot(x_line, intercept + slope * x_line, color="red", linewidth=2)

# Thêm công thức hồi quy
formula_text = f"y = {intercept:.2f} + ({slope:.2f}) * x"
plt.text(3.8, 30, formula_text, color="red")
plt.show()
```

### 9.5.5 Phân nhóm theo màu

```python
# Phân biệt xe theo số xy-lanh
colors = {4: "blue", 6: "magenta", 8: "gray"}

for cyl, group in mtcars.groupby("cyl"):
    plt.scatter(group["wt"], group["mpg"], color=colors[cyl], s=60,
                label=f"{cyl} cyl")   # label dùng cho chú thích

plt.title("MPG vs Trọng lượng (theo xy-lanh)")
plt.xlabel("Trọng lượng")
plt.ylabel("MPG")
plt.legend(loc="upper right")   # Thêm chú thích
plt.show()
```

### 9.5.6 Scatter Plot với nhiều biến

```python
# Ma trận scatter plot (R: pairs)
pd.plotting.scatter_matrix(mtcars[["mpg", "wt", "hp", "qsec"]],
                           figsize=(8, 8), color="blue", diagonal="hist")
plt.suptitle("Ma trận Scatter Plot")
plt.show()
```

**Giải thích:**
- Mỗi ô hiển thị mối quan hệ giữa 2 biến
- Đường chéo là histogram của từng biến (R hiển thị tên biến)
- Giúp nhìn tổng quan mối quan hệ giữa nhiều biến

---

## 9.6 Biểu đồ đường (Line Plot)

### 9.6.1 Line Plot cơ bản

**Line plot** thường dùng cho **dữ liệu chuỗi thời gian**

```python
# Ví dụ: Doanh thu theo tháng
months = np.arange(1, 13)
revenue = [50, 55, 60, 58, 65, 70, 75, 80, 78, 85, 90, 95]

plt.plot(months, revenue)   # plt.plot mặc định vẽ đường (R: type = "l")
plt.show()
```

### 9.6.2 Tùy chỉnh Line Plot

```python
plt.plot(months, revenue, color="blue", linewidth=2)   # Độ dày đường
plt.title("Doanh thu theo tháng")
plt.xlabel("Tháng")
plt.ylabel("Doanh thu (triệu đồng)")
plt.ylim(0, 100)

# Thêm điểm (R: points)
plt.scatter(months, revenue, color="red", zorder=3)   # zorder=3: vẽ điểm đè lên đường
plt.show()
```

### 9.6.3 Nhiều đường trên cùng biểu đồ

```python
# Doanh thu 2 chi nhánh
revenue_A = [50, 55, 60, 58, 65, 70, 75, 80, 78, 85, 90, 95]
revenue_B = [45, 50, 58, 62, 68, 72, 70, 75, 80, 82, 88, 92]

plt.plot(months, revenue_A, color="blue", linewidth=2, label="Chi nhánh A")
# Thêm đường thứ 2: chỉ cần gọi plt.plot thêm lần nữa (R: lines)
plt.plot(months, revenue_B, color="red", linewidth=2, label="Chi nhánh B")

plt.title("So sánh doanh thu 2 chi nhánh")
plt.xlabel("Tháng")
plt.ylabel("Doanh thu (triệu đồng)")
plt.ylim(40, 100)
plt.legend(loc="upper left")   # Thêm chú thích
plt.show()
```

### 9.6.4 Kết hợp đường và điểm

```python
plt.plot(months, revenue_A, color="darkgreen", linewidth=2,
         marker="o")   # Thêm marker → vừa đường vừa điểm (R: type = "b")
plt.title("Doanh thu chi nhánh A")
plt.xlabel("Tháng")
plt.ylabel("Doanh thu")
plt.show()
```

---

## 9.7 Biểu đồ tròn (Pie Chart)

### 9.7.1 Pie Chart cơ bản

```python
# Ví dụ: Phân bố sinh viên theo khoa
faculties = ["CNTT", "Kinh tế", "Ngoại ngữ", "Cơ khí"]
students = [450, 320, 280, 350]

plt.pie(students, labels=faculties)
plt.show()
```

### 9.7.2 Tùy chỉnh Pie Chart

```python
plt.pie(students, labels=faculties,
        colors=plt.cm.rainbow(np.linspace(0, 1, 4)))   # Màu cầu vồng
plt.title("Phân bố sinh viên theo khoa")
plt.show()

# Thêm phần trăm: matplotlib tự tính với autopct (không cần tự tính như R)
plt.pie(students, labels=faculties,
        autopct="%1.1f%%",                 # Hiển thị % với 1 chữ số thập phân
        startangle=90,                     # Bắt đầu từ góc 12 giờ
        colors=["lightblue", "lightcoral", "lightgreen", "lightyellow"])
plt.title("Phân bố sinh viên theo khoa")
plt.axis("equal")                          # Đảm bảo hình tròn
plt.show()
```

### 9.7.3 Lưu ý khi sử dụng Pie Chart

**Pie chart tốt cho:**
- Hiển thị tỷ lệ phần trăm
- Số lượng nhóm ít (3-5 nhóm)

**Không nên dùng khi:**
- Nhiều hơn 5-6 nhóm
- Cần so sánh chính xác giữa các nhóm
- Trong trường hợp này nên dùng **Bar Chart**

```python
# So sánh: chia hình làm 1 hàng, 2 cột (R: par(mfrow = c(1, 2)))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
rainbow4 = plt.cm.rainbow(np.linspace(0, 1, 4))

axes[0].pie(students, labels=faculties, colors=rainbow4)
axes[0].set_title("Pie Chart")

axes[1].bar(faculties, students, color=rainbow4)
axes[1].set_title("Bar Chart")
axes[1].set_ylabel("Số sinh viên")

plt.tight_layout()
plt.show()
```

---

## 9.8 Vẽ nhiều biểu đồ cùng lúc

### 9.8.1 Chia khung hình với plt.subplots()

`plt.subplots(nrows, ncols)` trả về `fig` (toàn bộ khung hình) và `axes` (mảng các ô biểu đồ) – tương đương `par(mfrow = c(nrows, ncols))` trong R. Mỗi ô được vẽ bằng các phương thức của `ax`: `ax.hist()`, `ax.boxplot()`, `ax.scatter()`, `ax.bar()`, `ax.set_title()`, ...

```python
# Chia thành 2 hàng, 2 cột (tổng 4 ô)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Vẽ 4 biểu đồ: axes[hàng, cột]
axes[0, 0].hist(mtcars["mpg"], color="lightblue", edgecolor="white")
axes[0, 0].set_title("Histogram: MPG")

axes[0, 1].boxplot(mtcars["mpg"])
axes[0, 1].set_title("Boxplot: MPG")

axes[1, 0].scatter(mtcars["wt"], mtcars["mpg"])
axes[1, 0].set_title("Scatter: MPG vs WT")

cyl_counts = mtcars["cyl"].value_counts().sort_index()
axes[1, 1].bar(cyl_counts.index.astype(str), cyl_counts.values, color="coral")
axes[1, 1].set_title("Bar: Cylinders")

plt.tight_layout()   # Tự căn chỉnh khoảng cách giữa các ô
plt.show()
```

> Không cần "trở về chế độ 1 biểu đồ" như `par(mfrow = c(1, 1))` của R – mỗi lần gọi `plt.subplots()` sẽ tạo một khung hình mới.

### 9.8.2 Ví dụ thực tế: Phân tích một biến

```python
# Phân tích toàn diện biến mpg
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 1. Histogram
axes[0, 0].hist(mtcars["mpg"], bins=10, color="skyblue", edgecolor="white")
axes[0, 0].set(title="Phân phối MPG", xlabel="MPG")

# 2. Boxplot
axes[0, 1].boxplot(mtcars["mpg"], patch_artist=True,
                   boxprops=dict(facecolor="lightgreen"))
axes[0, 1].set(title="Boxplot MPG", ylabel="MPG")

# 3. Boxplot theo nhóm
groups = [g["mpg"].values for _, g in mtcars.groupby("cyl")]
bp = axes[1, 0].boxplot(groups, tick_labels=[4, 6, 8], patch_artist=True)
for box, c in zip(bp["boxes"], ["red", "green", "blue"]):
    box.set_facecolor(c)
axes[1, 0].set(title="MPG theo Cylinders", xlabel="Cylinders", ylabel="MPG")

# 4. Scatter với biến khác + đường hồi quy
axes[1, 1].scatter(mtcars["wt"], mtcars["mpg"], color="darkblue")
slope, intercept = np.polyfit(mtcars["wt"], mtcars["mpg"], 1)
xs = np.linspace(mtcars["wt"].min(), mtcars["wt"].max(), 100)
axes[1, 1].plot(xs, intercept + slope * xs, color="red", linewidth=2)
axes[1, 1].set(title="MPG vs Weight", xlabel="Weight", ylabel="MPG")

plt.tight_layout()
plt.show()
```

> `ax.set(title=..., xlabel=..., ylabel=...)` là cách viết gọn để đặt nhiều thuộc tính cùng lúc.

### 9.8.3 Chia khung hình không đều

```python
# Tạo layout tùy chỉnh (R: layout(matrix(c(1, 1, 2, 3), nrow = 2, byrow = TRUE)))
fig, axd = plt.subplot_mosaic([["top", "top"],
                               ["left", "right"]], figsize=(9, 7))

# Biểu đồ 1: Chiếm 2 ô trên
axd["top"].hist(mtcars["mpg"], color="lightblue", edgecolor="white")
axd["top"].set_title("Histogram lớn")

# Biểu đồ 2: Ô dưới trái
axd["left"].boxplot(mtcars["mpg"], patch_artist=True,
                    boxprops=dict(facecolor="lightgreen"))
axd["left"].set_title("Boxplot nhỏ")

# Biểu đồ 3: Ô dưới phải
axd["right"].bar(cyl_counts.index.astype(str), cyl_counts.values, color="coral")
axd["right"].set_title("Barplot nhỏ")

plt.tight_layout()
plt.show()
```

---

## 9.9 Tùy chỉnh nâng cao

### 9.9.1 Màu sắc trong matplotlib

```python
# Màu cơ bản: dùng tên màu, mã hex, hoặc bộ (R, G, B)
colors_basic = ["red", "blue", "green", "yellow", "orange",
                "purple", "pink", "brown", "gray", "black"]
# Ví dụ khác: "#1f77b4", (0.2, 0.4, 0.6)

# Xem tất cả màu có tên: matplotlib.colors.CSS4_COLORS (khoảng 150 màu)

# Một số bảng màu (colormap) phổ biến – tương đương rainbow(), heat.colors()... của R
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
cmaps = [("rainbow", "Rainbow"), ("hot", "Heat Colors"),
         ("terrain", "Terrain Colors"), ("viridis", "Viridis")]

for ax, (cmap, title) in zip(axes.flat, cmaps):
    ax.bar(range(1, 6), range(1, 6), color=plt.get_cmap(cmap)(np.linspace(0, 0.9, 5)))
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

### 9.9.2 Thêm văn bản và chú thích

```python
plt.scatter(mtcars["wt"], mtcars["mpg"], color="blue")
plt.title("MPG vs Weight")
plt.xlabel("Weight (1000 lbs)")
plt.ylabel("MPG")

# Thêm text + mũi tên chỉ điểm trong một lệnh (R: text + arrows)
plt.annotate("Xe nặng\ntiêu tốn nhiều xăng",
             xy=(5.3, 15),          # Điểm mũi tên chỉ tới
             xytext=(4.5, 28),      # Vị trí đặt chữ
             color="red",
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.show()
```

### 9.9.3 Thêm lưới (grid)

```python
plt.scatter(mtcars["wt"], mtcars["mpg"], color="blue")
plt.title("MPG vs Weight (có lưới)")
plt.xlabel("Weight")
plt.ylabel("MPG")

# Thêm lưới
plt.grid(color="gray", linestyle=":")
plt.show()
```

### 9.9.4 Tùy chỉnh trục

```python
plt.scatter(mtcars["wt"], mtcars["mpg"], color="blue")
plt.title("MPG vs Weight")
plt.xlabel("Weight (1000 lbs)")
plt.ylabel("Miles per Gallon")
plt.xlim(0, 6)                 # Giới hạn trục x
plt.ylim(0, 40)                # Giới hạn trục y
plt.xticks(rotation=45)        # Xoay nhãn trục x (R: las)
plt.show()
```

### 9.9.5 Lưu biểu đồ ra file

Trong R phải mở thiết bị (`png()`), vẽ, rồi đóng (`dev.off()`). Trong matplotlib chỉ cần gọi `savefig()` **trước** `plt.show()`:

```python
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(mtcars["wt"], mtcars["mpg"], color="blue")
ax.set_title("MPG vs Weight")

# Lưu dưới dạng PNG
fig.savefig("my_plot.png", dpi=100)                       # 8x6 inch * 100 dpi = 800x600 pixel

# Lưu dưới dạng PDF (dạng vector, phóng to không vỡ – tốt cho in ấn, bài báo)
fig.savefig("my_plot.pdf")

# Lưu dưới dạng JPEG
fig.savefig("my_plot.jpg", dpi=100, pil_kwargs={"quality": 95})

# bbox_inches="tight": cắt bỏ khoảng trắng thừa quanh biểu đồ
fig.savefig("my_plot_tight.png", dpi=300, bbox_inches="tight")
plt.show()
```

---

## 9.10 Lựa chọn biểu đồ phù hợp

### HƯỚNG DẪN CHỌN BIỂU ĐỒ:

#### 1. Dữ liệu PHÂN LOẠI (categorical):
- **Một biến**: Bar Chart hoặc Pie Chart
- **Hai biến**: Grouped Bar Chart hoặc Stacked Bar Chart

#### 2. Dữ liệu LIÊN TỤC (continuous):
- **Phân phối một biến**: Histogram hoặc Box Plot
- **So sánh nhiều nhóm**: Box Plot
- **Mối quan hệ hai biến**: Scatter Plot
- **Xu hướng theo thời gian**: Line Plot

#### 3. KẾT HỢP:
- Scatter Plot + Line (xu hướng)
- Histogram + Density curve

### Ví dụ minh họa

```python
# Tạo dữ liệu mẫu
rng = np.random.default_rng(123)
category = rng.choice(["A", "B", "C"], size=100)       # R: sample(..., replace = TRUE)
continuous = rng.normal(50, 10, size=100)
time_series = np.cumsum(rng.normal(0, 5, size=50))

df = pd.DataFrame({"category": category, "continuous": continuous})
rainbow3 = plt.cm.rainbow(np.linspace(0, 1, 3))

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 1. Categorical: Bar Chart
counts = df["category"].value_counts().sort_index()
axes[0, 0].bar(counts.index, counts.values, color=rainbow3)
axes[0, 0].set_title("Phân loại - Bar Chart")

# 2. Continuous: Histogram
axes[0, 1].hist(continuous, color="lightblue", edgecolor="white")
axes[0, 1].set_title("Liên tục - Histogram")

# 3. Categorical + Continuous: Boxplot
groups = [g["continuous"].values for _, g in df.groupby("category")]
bp = axes[1, 0].boxplot(groups, tick_labels=counts.index, patch_artist=True)
for box, c in zip(bp["boxes"], rainbow3):
    box.set_facecolor(c)
axes[1, 0].set_title("So sánh nhóm - Boxplot")

# 4. Time series: Line Plot
axes[1, 1].plot(time_series, color="blue", linewidth=2)
axes[1, 1].set_title("Chuỗi thời gian - Line")

plt.tight_layout()
plt.show()
```

---

## BÀI TẬP THỰC HÀNH

### Bài tập 1: Bar Chart

Sử dụng dữ liệu sau:
```python
subjects = ["Toán", "Lý", "Hóa", "Văn", "Anh"]
scores = [8, 7.5, 9, 8.5, 7]
```

**Yêu cầu:**
1. Vẽ bar chart cơ bản
2. Thêm tiêu đề "Điểm thi của bạn"
3. Tô màu khác nhau cho mỗi môn
4. Thêm giá trị điểm lên đầu mỗi cột (gợi ý: `ax.bar_label`)
5. Vẽ bar chart ngang (`barh`)

### Bài tập 2: Histogram

Tạo dữ liệu: Điểm thi của 100 sinh viên
```python
rng = np.random.default_rng(2024)
exam_scores = rng.normal(loc=70, scale=10, size=100)
```

**Yêu cầu:**
1. Vẽ histogram với 10 bins
2. Thêm tiêu đề và nhãn trục phù hợp
3. Tô màu xanh lam
4. Thêm đường thẳng đứng màu đỏ tại vị trí điểm trung bình (`axvline`)
5. Vẽ histogram khác với 20 bins, so sánh sự khác biệt

### Bài tập 3: Box Plot

Sử dụng dữ liệu iris

**Yêu cầu:**
1. Vẽ box plot so sánh `Petal.Length` giữa 3 loài
2. Tô màu khác nhau cho mỗi loài
3. Thêm tiêu đề phù hợp
4. Nhìn vào biểu đồ và trả lời:
   - Loài nào có petal dài nhất?
   - Loài nào có độ biến thiên lớn nhất?
   - Có outliers không? Ở loài nào?

### Bài tập 4: Scatter Plot

Sử dụng dữ liệu mtcars

**Yêu cầu:**
1. Vẽ scatter plot giữa hp (horsepower) và mpg
2. Tô màu các điểm theo số xy-lanh (cyl)
3. Thêm đường hồi quy tuyến tính (`np.polyfit`)
4. Thêm legend giải thích màu
5. Nhận xét về mối quan hệ giữa hp và mpg

### Bài tập 5: Nhiều biểu đồ

Sử dụng dữ liệu mtcars

**Yêu cầu:**

Tạo một figure với 4 biểu đồ (2x2) để phân tích biến hp:
1. Histogram của hp
2. Box plot của hp
3. Box plot so sánh hp theo cyl
4. Scatter plot hp vs mpg

### Bài tập 6: Tổng hợp

Tạo dữ liệu bán hàng của 4 quý:
```python
sales = pd.DataFrame({
    "Q1": [100, 120, 110, 130],
    "Q2": [150, 140, 160, 155],
    "Q3": [180, 170, 190, 185],
    "Q4": [200, 210, 195, 220],
}, index=["Sản phẩm A", "Sản phẩm B", "Sản phẩm C", "Sản phẩm D"])
```

**Yêu cầu:**
1. Vẽ grouped bar chart so sánh doanh thu 4 quý
2. Vẽ line plot cho từng sản phẩm qua 4 quý (gợi ý: `sales.T.plot()`)
3. Tính tổng doanh thu mỗi quý, vẽ bar chart
4. Tạo figure 2x2 hiển thị:
   - Grouped bar chart
   - Line plot tất cả sản phẩm
   - Pie chart tổng doanh thu mỗi quý
   - Bar chart tổng doanh thu mỗi sản phẩm

---

## TÀI LIỆU THAM KHẢO

1. **Matplotlib – Quick start guide**: https://matplotlib.org/stable/users/explain/quick_start.html
2. **Matplotlib – Plot types gallery**: https://matplotlib.org/stable/plot_types/index.html
3. **pandas – Chart visualization**: https://pandas.pydata.org/docs/user_guide/visualization.html
4. **Seaborn** (thư viện vẽ thống kê nâng cao, xây dựng trên matplotlib): https://seaborn.pydata.org/

---

## TỔNG KẾT

### Những điểm cần nhớ:

1. ✅ **Chọn biểu đồ phù hợp với loại dữ liệu:**
   - **Bar chart** (`bar`, `barh`): Dữ liệu phân loại
   - **Histogram** (`hist`): Phân phối dữ liệu liên tục
   - **Box plot** (`boxplot`): So sánh nhóm, tìm outliers
   - **Scatter plot** (`scatter`): Mối quan hệ giữa 2 biến
   - **Line plot** (`plot`): Xu hướng theo thời gian
   - **Pie chart** (`pie`): Tỷ lệ phần trăm (ít nhóm)

2. ✅ **Luôn thêm tiêu đề và nhãn trục rõ ràng** (`title`, `xlabel`, `ylabel`)

3. ✅ **Sử dụng màu sắc hợp lý:**
   - Không quá nhiều màu
   - Màu có ý nghĩa (đỏ = cảnh báo, xanh lá = tốt)
   - Đảm bảo đọc được khi in đen trắng

4. ✅ **Box plot giúp:**
   - Thấy trung vị, Q1, Q3
   - Phát hiện outliers
   - So sánh nhiều nhóm

5. ✅ **Histogram vs Bar chart:**
   - **Histogram**: Dữ liệu liên tục, không có khoảng cách giữa cột
   - **Bar chart**: Dữ liệu phân loại, có khoảng cách

6. ✅ **Sử dụng `plt.subplots(nrows, ncols)` để vẽ nhiều biểu đồ cùng lúc**

7. ✅ **Lưu biểu đồ**: `fig.savefig("ten_file.png")` (png, pdf, jpg, svg...)

### Bảng đối chiếu nhanh R → Python

| R (base graphics) | matplotlib |
| :--- | :--- |
| `barplot()` | `plt.bar()`, `plt.barh()`, `df.plot(kind="bar")` |
| `hist()` | `plt.hist()` |
| `boxplot()` | `plt.boxplot()`, `df.boxplot()` |
| `plot(x, y)` | `plt.scatter(x, y)` |
| `plot(x, y, type = "l")` | `plt.plot(x, y)` |
| `pie()` | `plt.pie()` |
| `pairs()` | `pd.plotting.scatter_matrix()` |
| `main`, `xlab`, `ylab` | `plt.title()`, `plt.xlabel()`, `plt.ylabel()` |
| `col`, `pch`, `cex`, `lwd`, `lty` | `color`, `marker`, `s`, `linewidth`, `linestyle` |
| `abline(h =, v =)` | `plt.axhline()`, `plt.axvline()` |
| `lines()`, `points()`, `text()` | `plt.plot()`, `plt.scatter()`, `plt.text()` |
| `legend()` | `plt.legend()` |
| `par(mfrow = c(2, 2))` | `fig, axes = plt.subplots(2, 2)` |
| `png()` ... `dev.off()` | `fig.savefig("file.png")` |

### Quy trình vẽ biểu đồ tốt:

1. **Xác định mục đích**: Muốn truyền đạt thông tin gì?
2. **Chọn loại biểu đồ** phù hợp
3. **Vẽ biểu đồ cơ bản**
4. **Thêm tiêu đề, nhãn, màu sắc**
5. **Kiểm tra** xem biểu đồ có dễ hiểu không
6. **Lưu lại** nếu cần

### Lưu ý quan trọng:

- Biểu đồ phải **đơn giản, dễ hiểu**
- **Không** thêm quá nhiều thông tin vào một biểu đồ
- Luôn nghĩ về **người xem**
- **"A picture is worth a thousand words"** - Một hình ảnh đáng giá ngàn lời

---

**Cập nhật**: Tháng 3/2026
