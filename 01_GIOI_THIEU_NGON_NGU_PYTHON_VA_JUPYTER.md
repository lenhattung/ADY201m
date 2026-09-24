# BÀI 1: GIỚI THIỆU NGÔN NGỮ PYTHON VÀ MÔI TRƯỜNG LÀM VIỆC (JUPYTER / VS CODE)

## 1. Giới thiệu chung về ngôn ngữ Python
Python là một ngôn ngữ lập trình **đa năng**, mã nguồn mở, cú pháp gần với ngôn ngữ tự nhiên. Trong vài năm gần đây, Python đã trở thành ngôn ngữ phổ biến nhất cho **Khoa học Dữ liệu (Data Science)**, **Thống kê** và **Học máy (Machine Learning)**.

* **Nguồn gốc:** Được phát triển bởi Guido van Rossum tại Viện CWI (Hà Lan), phát hành lần đầu năm 1991.
* **Vị thế:** Là công cụ tiêu chuẩn trong phân tích dữ liệu, học máy, trí tuệ nhân tạo (AI) và tự động hóa.
* **Đặc điểm:**
    * **Mã nguồn mở:** Hoàn toàn miễn phí.
    * **Hệ sinh thái khổng lồ:** Hàng trăm nghìn gói thư viện (packages) trên kho **PyPI** (Python Package Index).
    * **Cộng đồng:** Cộng đồng rất lớn, tài liệu phong phú, hỗ trợ tốt từ cơ bản đến nghiên cứu chuyên sâu.

## 2. Tại sao nên sử dụng Python cho Khoa học Dữ liệu?
| Đặc điểm | Lợi ích |
| :--- | :--- |
| **Thống kê** | `scipy.stats`, `statsmodels` cung cấp các kiểm định và mô hình thống kê từ cổ điển đến hiện đại. |
| **Trực quan hóa** | `matplotlib`, `seaborn` giúp tạo biểu đồ chất lượng cao (đủ tiêu chuẩn in sách báo). |
| **Xử lý dữ liệu** | `numpy` và `pandas` xử lý mạnh mẽ các bộ dữ liệu lớn và phức tạp. |
| **Học máy / AI** | `scikit-learn`, `PyTorch`, `TensorFlow` là các thư viện hàng đầu thế giới. |
| **Báo cáo** | Jupyter Notebook kết hợp code + kết quả + văn bản, xuất ra HTML, PDF trực tiếp. |

### Bộ thư viện cốt lõi sẽ dùng trong môn học

| Thư viện | Vai trò | Tương đương trong R |
| :--- | :--- | :--- |
| `numpy` | Mảng số, tính toán vector hóa | vector, matrix |
| `pandas` | Bảng dữ liệu (DataFrame), làm sạch dữ liệu | data.frame, dplyr |
| `matplotlib` | Vẽ biểu đồ | base graphics (`plot`, `hist`, ...) |
| `scipy` | Thống kê, xác suất | `stats` (`dnorm`, `t.test`, ...) |
| `scikit-learn` | Học máy, phân cụm | `kmeans`, `cluster`, `dbscan` |

## 3. Phân biệt Python và Môi trường lập trình (IDE)
Một sai lầm phổ biến của người mới bắt đầu là nhầm lẫn giữa hai khái niệm này:
* **Python (The Engine):** Là "bộ não" – trình thông dịch (interpreter) thực hiện các phép tính và xử lý lệnh.
* **Jupyter Notebook / JupyterLab / VS Code (The Interface):** Là phần mềm giao diện giúp chúng ta viết code Python dễ dàng hơn, quản lý file và xem biểu đồ trực quan hơn.

> **Ví dụ:** Nếu Python là động cơ của một chiếc xe, thì Jupyter/VS Code chính là khoang lái với vô lăng và bảng điều khiển.

---

## 4. Hướng dẫn cài đặt chi tiết

### Cách 1 (Khuyên dùng cho người mới): Cài đặt Anaconda
Anaconda là bộ cài "trọn gói" gồm Python + Jupyter + hầu hết thư viện Data Science (numpy, pandas, matplotlib, scipy, scikit-learn...).

1.  Truy cập website chính thức: [https://www.anaconda.com/download](https://www.anaconda.com/download)
2.  Chọn phiên bản phù hợp với hệ điều hành của bạn (Windows, macOS, hoặc Linux).
3.  Chạy file cài đặt với các tùy chọn mặc định (nhấn *Next* liên tục).
4.  Mở **Anaconda Navigator** → chọn **JupyterLab** (hoặc **Jupyter Notebook**) → **Launch**.

### Cách 2: Cài Python gốc + VS Code
1.  Tải Python tại: [https://www.python.org/downloads/](https://www.python.org/downloads/)
    * **Lưu ý (Windows):** Tích chọn ô **"Add python.exe to PATH"** trước khi nhấn *Install Now*.
2.  Tải VS Code tại: [https://code.visualstudio.com/](https://code.visualstudio.com/)
3.  Trong VS Code, cài 2 extension: **Python** và **Jupyter** (của Microsoft).
4.  Mở Terminal và cài các thư viện cần cho môn học:

```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn jupyterlab
```

### Cách 3: Không cần cài đặt – Google Colab
Truy cập [https://colab.research.google.com](https://colab.research.google.com), đăng nhập tài khoản Google và tạo Notebook mới. Colab đã có sẵn toàn bộ thư viện của môn học.

---

## 5. Làm quen với giao diện Jupyter (4 khu vực)

Sau khi mở JupyterLab, giao diện gồm các khu vực chính:

1.  **File Browser (Thanh bên trái):** Quản lý thư mục làm việc, tạo/mở file `.ipynb` (notebook) và `.py` (script).
2.  **Notebook (Khu vực giữa):** Gồm các **ô (cell)**:
    * *Code cell:* Nơi viết code. Nhấn `Shift + Enter` để chạy ô và chuyển xuống ô tiếp theo (`Ctrl + Enter` để chạy mà không chuyển ô).
    * *Markdown cell:* Nơi viết ghi chú, tiêu đề, công thức.
3.  **Output (Ngay dưới mỗi ô code):** Kết quả tính toán, bảng dữ liệu và biểu đồ hiển thị ngay bên dưới ô vừa chạy.
4.  **Variable Inspector / Kernel:** 
    * *Kernel:* Là "bộ não" Python đang chạy phía sau. Menu **Kernel → Restart** để xóa toàn bộ biến trong bộ nhớ.
    * *Variables* (trong VS Code: nút **Variables** trên thanh công cụ): Hiển thị các biến, bảng dữ liệu (DataFrame) đang được lưu trong bộ nhớ – tương tự cửa sổ *Environment* của RStudio.

> **Tra cứu hướng dẫn hàm:** Gõ `help(ten_ham)` hoặc `ten_ham?` (trong Jupyter) để xem tài liệu của hàm.

> **Mẹo Jupyter:** Nếu dòng cuối cùng của một ô là một biểu thức (ví dụ tên biến), Jupyter sẽ tự động hiển thị giá trị của nó mà không cần `print()`.

---

## 6. Bài tập kiểm tra đầu tiên
Hãy tạo một Notebook mới, gõ đoạn code sau vào một ô code và nhấn **Shift + Enter**:

```python
# Phép toán cơ bản
a = 10
b = 20
print(a + b)

# Kiểm tra phiên bản Python
import sys
print(sys.version)

# Kiểm tra các thư viện chính đã được cài đặt
import numpy as np
import pandas as pd
import matplotlib
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)
```

> **Quy ước import dùng trong suốt môn học:**
> ```python
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> ```
