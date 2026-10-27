# AIO Projects

Tập hợp bài thực hành và thử nghiệm AI/ML trong quá trình học AIO, cùng các
thử nghiệm phân tích và mô phỏng dữ liệu cảm biến. Mỗi thư mục là một dự án
riêng, có dữ liệu, môi trường và cách chạy khác nhau.

## Danh mục dự án

| Dự án | Nội dung | Điểm bắt đầu |
| --- | --- | --- |
| [Project-5.1](Project-5.1/) — Dự báo giá nhà | Tiền xử lý, feature engineering, hồi quy, so sánh mô hình và giao diện Streamlit. | [README](Project-5.1/README.md), [pipeline](Project-5.1/app.py), [bài viết](Project-5.1/Blog-House_Sale_Prediction.md) |
| [Project-6.1](Project-6.1/) — Dự báo chuỗi thời gian | Thử nghiệm Linear/DLinear/NLinear, RevIN và HMM cho dữ liệu giá cổ phiếu FPT. | [README](Project-6.1/README.md), [notebook chính](Project-6.1/FPT_LTSF_GridSearch_Extended.ipynb), [bài viết](Project-6.1/BLOG.md) |
| [Project-7.1](Project-7.1/) — Nhận dạng hành động trong video | So sánh ViT và VideoMAE trên HMDB51; ablation về augmentation, mixup, huấn luyện hai giai đoạn và test-time augmentation. | [README](Project-7.1/README.md), [notebook ablation](Project-7.1/notebooks/unified-ablation.ipynb), [phân tích baseline](Project-7.1/docs/baseline-btc-analysis.md) |
| [Contruction-Projects](Contruction-Projects/) — Phân tích cảm biến máy xúc | Notebook khám phá dữ liệu gia tốc, con quay hồi chuyển và nhãn hoạt động. | [EDA dữ liệu máy xúc](Contruction-Projects/notebooks/EDA_Construction_Sensor_Data.ipynb), [EDA cảm biến](Contruction-Projects/notebooks/EDA_Sensor_Data.ipynb) |
| [IMU-Synth-Generator](IMU-Synth-Generator/) — Dữ liệu IMU tổng hợp | Tạo CSV mô phỏng tín hiệu gia tốc và con quay hồi chuyển, tùy chỉnh ngày, giờ làm việc và random seed. | [README](IMU-Synth-Generator/README.md), [CLI](IMU-Synth-Generator/generate_excavator_data.py) |

## Bắt đầu

```bash
git clone https://github.com/Hirin/AIO-Projects.git
cd AIO-Projects
```

Chọn một dự án bên dưới và chạy lệnh từ đúng thư mục. Nên dùng môi trường
ảo riêng cho từng dự án; repository không có một bộ dependencies chung.
Các lệnh kích hoạt môi trường bên dưới dành cho Linux/macOS. Trên Windows
PowerShell, dùng `.venv\Scripts\Activate.ps1`.

### Tạo dữ liệu IMU — ví dụ chạy cục bộ

```bash
cd IMU-Synth-Generator
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_imu.txt "numpy==2.3.4"
python generate_excavator_data.py --help
python generate_excavator_data.py --date 2025-08-15 --days 1 --seed 42 --output sample.csv
```

File dependencies hiện có là `requirements_imu.txt`. Ví dụ trên đã chạy với
Python 3.14, NumPy 2.3.4 và pandas 3.1.0. Giữ phiên bản NumPy này cho ví dụ:
code generator hiện gặp lỗi gán mảng vào phần tử scalar với NumPy 2.5.4.
Kết quả `sample.csv`
là dữ liệu tổng hợp để thử nghiệm; cần đánh giá riêng trước khi dùng thay cho
dữ liệu đo thực tế.

### Project-5.1 — pipeline giá nhà

Dùng Python 3.12 trở lên theo [pyproject.toml](Project-5.1/pyproject.toml).
Từ thư mục gốc repository:

```bash
cd Project-5.1
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py --help
```

Chuẩn bị tập huấn luyện **House Prices: Advanced Regression Techniques** từ
nguồn được dẫn trong [README dự án](Project-5.1/README.md), rồi đặt tại
`Project-5.1/data/raw/train-house-prices-advanced-regression-techniques.csv`.
File dữ liệu gốc này không có trong checkout; cần tự chuẩn bị trước khi chạy:

```bash
python app.py --step all
python -m streamlit run streamlit_app.py
```

Pipeline ghi kết quả vào `data/processed/`, `data/interim/` và `models/`.
Giao diện Streamlit cần model cùng các artifact xử lý tương ứng; chỉ cài
dependencies chưa đủ để chạy dự báo. Xem [Serving.py](Project-5.1/src/Serving.py)
để đối chiếu các file cần thiết.

### Project-6.1 — notebook dự báo FPT

```bash
cd Project-6.1
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab torch pandas numpy scikit-learn hmmlearn matplotlib seaborn tqdm
python -m jupyter lab FPT_LTSF_GridSearch_Extended.ipynb
```

Chọn kernel của môi trường vừa tạo và chạy từ `Project-6.1/` để đường dẫn
`data/FPT_train.csv` trong notebook trỏ đúng tới dữ liệu đã có. Notebook thực
hiện grid search và ghi dự báo vào `submissions/`; kiểm tra cấu hình trước
khi chạy toàn bộ vì thời gian huấn luyện phụ thuộc cấu hình và phần cứng.

### Project-7.1 và notebook cảm biến

- **Video action recognition:** mở
  [unified-ablation.ipynb](Project-7.1/notebooks/unified-ablation.ipynb) trên
  Kaggle, chuẩn bị dữ liệu và chỉnh `PATH_DATA_TRAIN` / `PATH_DATA_TEST`
  theo vị trí thực tế. Notebook hiện dùng đường dẫn `/kaggle/input/action-video/`;
  dữ liệu video không được lưu trong repository. Bắt đầu với `WARMUP = True`
  để kiểm tra pipeline trước khi huấn luyện. Xem dependencies và các toggle
  trong [README](Project-7.1/README.md).
- **Cảm biến máy xúc:** mở notebook trong `Contruction-Projects/notebooks/`,
  kiểm tra đường dẫn CSV và cài các thư viện được import ở đầu notebook.
  Thư mục này chưa có manifest dependencies riêng.

## Phạm vi và nguồn tham khảo

Đây là nơi lưu bài thực hành, notebook, thử nghiệm và báo cáo học tập. Các kết
quả trong từng dự án gắn với dữ liệu, cách chia tập và cấu hình được ghi ở đó.

- [Bài viết Project-6.1](Project-6.1/BLOG.md) ghi rõ bối cảnh cuộc thi
  **AIO-2025: LTSF-Linear Forecasting Challenge** và dẫn các tài liệu tham khảo.
- Project-7.1 giữ [baseline của ban tổ chức](Project-7.1/notebooks/baseline-btc.ipynb)
  cùng [phân tích so sánh](Project-7.1/docs/comparison-baseline-vs-phase3.md)
  để phân biệt baseline với các thử nghiệm mở rộng.
- Notebook cảm biến ghi nguồn dữ liệu **Becker & Keefe (2022)**. Bộ IMU
  generator tạo dữ liệu tổng hợp và được tách thành thư mục riêng.
- Đề bài, dữ liệu, baseline và mô hình pretrained có nguồn gốc riêng; xem
  attribution và điều kiện sử dụng tại tài liệu của từng dự án/nguồn dữ liệu.

Python bytecode (`__pycache__/`, `*.pyc`) và môi trường ảo là file sinh cục bộ,
không cần đưa vào Git.
