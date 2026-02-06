# I. Vision Transformer (ViT)

Trước tiên khi đi vào sâu tìm hiểu về model VideoMAE thì nhóm sẽ review sơ về cách hoạt động của model ViT:

![](images/Pasted%20image%2020260205201425.png)

## 1. Chia nhỏ hình ảnh thành các "Patch"

Thay vì xử lý từng pixel (rất tốn kém về tính toán), ViT chia hình ảnh thành các ô vuông nhỏ gọi là **Patches**.
- Ví dụ: Một ảnh $224 \times 224$ có thể được chia thành các patch kích thước $16 \times 16$.
- Mỗi patch này được coi như một "từ" (token) trong một câu văn.

![](images/Pasted%20image%2020260205202102.png)

## 2. Linear Projection và Position Embedding

- **Linear Projection:** Các patch (vốn là mảng 2D) sẽ được "trải phẳng" thành các vector 1D và đi qua một lớp tuyến tính để chuyển đổi thành các **Embedded Patches**.
- **Position Embedding:** Vì cơ chế Attention không biết thứ tự của các mảnh, ViT cộng thêm một bộ mã vị trí (các số $0, 1, 2, ...$ trong hình) vào mỗi vector. Điều này giúp mô hình biết mảnh nào nằm ở góc trên bên trái, mảnh nào nằm ở giữa.
- **[class] token:** Một vector đặc biệt (ký hiệu là $0$ trong hình) được thêm vào đầu chuỗi. Vector này sẽ "thu thập" thông tin từ tất cả các patch khác để dùng cho việc phân loại cuối cùng.

![](images/Pasted%20image%2020260205202300.png)

## 3. Transformer Encoder (Trái tim của ViT)

Toàn bộ các vector này được đưa vào các lớp **Transformer Encoder**. Bên trong gồm:
- **Multi-Head Attention:** Giúp các patch "giao tiếp" với nhau. Ví dụ: patch chứa "tai mèo" sẽ chú ý đến patch chứa "mắt mèo" để hiểu được cấu trúc tổng thể.
- **MLP (Multi-Layer Perceptron):** Các lớp mạng nơ-ron truyền thống để xử lý đặc trưng sau khi đã qua Attention.
- **Layer Norm & Residual Connection:** Giúp việc huấn luyện ổn định và hiệu quả hơn.

## 4. MLP Head và Phân loại

Sau khi đi qua nhiều lớp Encoder, thông tin cuối cùng của **[class] token** sẽ được đưa vào một bộ phân loại gọi là **MLP Head**. Tại đây, mô hình sẽ đưa ra kết quả dự đoán cuối cùng: đây là Chim, Mèo hay Chó.

![](images/Pasted%20image%2020260205202647.png)

---

# II. ImageMAE

Tiếp theo là sẽ tìm hiểu về cách hoạt động của ImageMAE, cũng dùng ViT làm backbone và là nền móng để phát triển thành VideoMAE:

![](images/Pasted%20image%2020260205203644.png)
![](images/Pasted%20image%2020260205203832.png)

## 1. Cơ chế Masking

Bước đầu tiên cũng vẫn là chia hình ảnh thành các mảnh nhỏ (patches). Tuy nhiên, thay vì giữ nguyên, ImageMAE sẽ **che đi (mask)** một phần rất lớn của bức ảnh.
- **Tỷ lệ che cực cao:** Thường là **75%** hoặc hơn (như trong hình bạn gửi là chỉ giữ lại **< 25%** mảnh nhìn thấy).    
- Việc che này tạo ra một "bài toán" khó cho AI: Phải nhìn một vài mảnh nhỏ (ví dụ: chỉ thấy cái mỏ chim) và đoán xem toàn bộ con chim trông như thế nào.

## 2. Bộ mã hóa Encoder (Chỉ xử lý phần "Visible")

Đây là điểm thông minh nhất của ImageMAE:
- **ViT làm Backbone:** Encoder chính là một khối **Vision Transformer (ViT)**.
- **Tiết kiệm tài nguyên:** Khác với các mô hình thông thường xử lý toàn bộ ảnh, Encoder của MAE **chỉ chạy trên những mảnh không bị che**. Vì chỉ phải xử lý 25% dữ liệu, tốc độ huấn luyện nhanh hơn và ít tốn RAM hơn rất nhiều.
- Kết quả đầu ra của bước này là các **Latent representation** (mã hóa đặc trưng) của các mảnh nhìn thấy.

## 3. Bộ giải mã Decoder và Reconstruction

Sau khi có đặc trưng từ Encoder, mô hình sẽ thực hiện bước "hồi phục":
- **Chèn Mask Tokens:** Mô hình lấy các đặc trưng từ Encoder và chèn thêm các "mảnh trống" (mask tokens) vào đúng vị trí của những phần đã bị che trước đó.

![](images/Pasted%20image%2020260205204424.png)

- **Decoder nhẹ:** Một bộ Decoder (thường nhỏ và nhẹ hơn Encoder) sẽ nhìn vào sự sắp xếp này và cố gắng vẽ lại (reconstruct) các pixel gốc cho những phần bị che.
- **Mục tiêu (Loss):** Mô hình sẽ so sánh ảnh được vẽ lại với ảnh gốc ban đầu bằng hàm **MSE Loss**. Nếu nó vẽ lại càng giống thật, nghĩa là nó đã hiểu sâu về cấu trúc của hình ảnh đó.

---

# III. VideoMAE

Món chính VideoMAE được phát triển lên từ ImageMAE:

![](images/Pasted%20image%2020260205204714.png)

## 1. Từ "Mảnh ảnh" (Patch) sang "Khối video" (Tubelet)

Trong ImageMAE, chúng ta chia ảnh 2D thành các ô vuông. Ở VideoMAE, chúng ta chia clip thành các khối lập phương (3D patches).
- Mỗi khối này bao gồm cả không gian (chiều rộng, chiều cao) và thời gian (số lượng khung hình).
- Ví dụ: Một khối có thể có kích thước $16 \times 16$ pixel và kéo dài qua $2$ khung hình liên tiếp.

## 2. Chiến lược Extreme Tube Masking

![](images/Pasted%20image%2020260205205431.png)

Đây là điểm khác biệt lớn nhất so với ImageMAE.
- **Tỷ lệ che cực lớn:** ImageMAE chỉ che 75%, nhưng VideoMAE che tới **90% - 95%**.
- **Tại sao?** Video có sự "dư thừa" thông tin rất lớn. Nếu bạn thấy một người đang bước đi ở giây thứ 1 và giây thứ 3, bạn dễ dàng đoán được họ ở đâu vào giây thứ 2. Nếu chỉ che 75%, AI sẽ "học vẹt" bằng cách nhìn các khung hình lân cận thay vì thực sự hiểu nội dung.
- **Frame Masking (b):** Che nguyên khung hình -> Quá khó hoặc mất liên kết thời gian.
- **Random Masking (c):** Che lố nhố -> AI dễ dàng "copy" pixel từ khung hình bên cạnh.
- **Tube Masking (d):** Để ngăn AI "gian lận", VideoMAE thường che theo trục thời gian (giống như 1 cái ống xuyên thẳng theo trục thời gian). Nếu một vị trí bị che ở khung hình đầu, nó sẽ bị che xuyên suốt cả clip. Điều này buộc AI phải vận dụng khả năng suy luận logic rất cao để tái tạo.

![](images/Pasted%20image%2020260205210819.png)

## 3. Encoder (ViT)

Giống hệt ImageMAE, phần **ViT** nằm ở đây:
- Nó chỉ nhận **10%** lượng dữ liệu không bị che (các visible tokens).
- Nhờ lượng dữ liệu đầu vào cực ít, ViT có thể xử lý các đoạn video dài mà không bị "nổ" bộ nhớ (RAM).
- ViT sẽ học cách liên kết các mảnh rời rạc này để hiểu: "À, mảnh này là tay người đang vung lên, mảnh kia là quả bóng đang bay".

## 4. Decoder và Reconstruction

Sau khi ViT xử lý xong, mô hình thực hiện:
- Ghép các mảnh đã xử lý với các "mảnh trống" (mask tokens).
- Đưa qua một Decoder nhẹ để "vẽ" lại toàn bộ video gốc.
- Sai số giữa video vẽ lại và video gốc chính là bài học để mô hình tự sửa mình.

## 5. Thông tin thực nghiệm (Paper Details)

### 5.1 Dataset Pretrain

- **K400 (Kinetics-400):** Quy mô ~306,245 video clip. Gồm 400 lớp hành động.
- **SSV2 (Something-Something V2):** Tập trung vào tương tác vật lý. Quy mô ~220,847 video. Gồm 174 lớp hành động.

### 5.2 Kết quả thực nghiệm

![](images/Pasted%20image%2020260205210140.png)

=> Đây là lý do nhóm dùng sẵn trọng số của VideoMAE trên dataset K400 để fine tune tiếp tục trên data Kaggle 7.1

---

# IV. Project 7.1

## 1. ViT-small (~22M params)
BTC đã áp dụng model này để làm baseline và có kết quả trên tập private test là 0.6922 

## 2. ViT-base (~86M params)
Nhóm thử nghiệm thêm trên model ViT lớn hơn là ViT-base và có kết quả trên tập private test nhỉnh hơn tý là 0.7373

## 3. VideoMAE (Ablation Study)
**Cấu hình mô hình:** Nhóm sử dụng VideoMAE với trọng số pretrained trên tập **Kinetics-400**, xây dựng trên backbone **ViT-base**. Do đặc thù xử lý video (thêm chiều thời gian), số lượng tham số xấp xỉ **86M**, tương đương với kiến trúc ViT tiêu chuẩn nhưng có khả năng trích xuất đặc trưng không gian - thời gian mạnh mẽ hơn.

### 3.1 Bảng kết quả thử nghiệm lũy kế (Incremental Ablation) với 20 epochs

| STT | Thử nghiệm (Increment) | Train Acc | Test Acc | Nhận xét |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | **Baseline (Vanilla)** | **100%** | **81.76%** | Overfitting nặng (Gap ~18%) |
| **Exp 2** | **+ Consistent Aug** | **100%** | **83.73%** | Giảm lỗi trích xuất đặc trưng |
| **Exp 3** | **+ Mixup (α=0.8)** | **89.98%** | **84.31%** | Cải thiện tính tổng quát (Regularization) |
| **Exp 4** | **+ 2-Stage & TTA** | **~91.00%** | **84.90%** | Tối ưu hóa xác thực và inference |

**=> Với 30 epochs từ final pipeline (Exp 4) thì score lên 85.10%**

### 3.2 Phân tích chi tiết các giai đoạn cải thiện

- **Exp 1 - Baseline (Overfitting nguyên bản):** Ở cấu hình mặc định, mô hình đạt **Train Acc 100%** nhưng **Test Acc chỉ dừng ở 81.76%**. Điều này cho thấy VideoMAE rất dễ "học vẹt" (memorization) trên tập dữ liệu Project 7.1 (vốn có quy mô nhỏ hơn K400), dẫn đến khả năng tổng quát hóa kém.
- **Exp 2 - Tác động của Consistent spatial augmentation:** Việc áp dụng cùng một phép biến đổi (Flip, Crop) cho toàn bộ 16 frames trong một video giúp Test Acc tăng lên **83.73%**. Tuy nhiên, mô hình vẫn đạt **Train Acc 100%**, cho thấy các phép tăng cường không gian đơn thuần chưa đủ sức ngăn cản sự hội tụ quá mức vào các đặc trưng cục bộ.
- **Exp 3 - Mixup và sự thay đổi về bản chất học:** Đây là bước ngoặt quan trọng nhất. Khi áp dụng Mixup, **Train Acc giảm từ 100% xuống còn ~90%**, trong khi **Test Acc tăng lên 84.31%**. Việc "ép" mô hình không được đạt độ chính xác tuyệt đối trên tập huấn luyện đã buộc nó phải học các ranh giới quyết định (decision boundaries) mềm mỏng và bền bỉ hơn, thay vì chỉ ghi nhớ các mẫu cụ thể.
- **Exp 4 - Chiến lược 2-Stage Training và TTA:**
    - **Phase 1 (Mixup):** Tạo nền tảng vững chắc với tính tổng quát cao.
    - **Phase 2 (Label Smoothing):** Làm mịn trọng số ở những epoch cuối với LR cực thấp, giúp mô hình hội tụ sâu hơn vào vùng cực tiểu toàn cục.
    - **TTA (6-View):** Việc đánh giá đa góc nhìn (3 spatial crops x 2 flip) giúp củng cố độ tin cậy của dự đoán, đưa kết quả cuối cùng đạt **84.90%**.

### 3.3 Đánh giá ảnh hưởng của tham số `num_frames` trong VideoMAE:

Mô hình VideoMAE được thiết lập mặc định với cấu hình pretrain là `num_frames = 16`. Đặc điểm này buộc mô hình luôn trích xuất đúng 16 khung hình từ mỗi video đầu vào để đưa vào quá trình huấn luyện/đánh giá, bất kể độ dài thực tế của video đó. Trong trường hợp video có ít hơn 16 khung hình, hệ thống sẽ tự động áp dụng phương pháp **Oversampling (Lấy mẫu lặp)** để đảm bảo tính nhất quán về kích thước đầu vào cho Transformer.

Nhận thấy tập dữ liệu thực tế (như trên Kaggle) có độ dài biến thiên khá lớn (dao động từ 5 đến 20 frames), nhóm đã thực hiện thử nghiệm giảm `num_frames` xuống còn 8 để tối ưu hóa tài nguyên. Kết quả thu được như sau:
- **Về hiệu năng:** Với cùng 30 epochs huấn luyện, cấu hình 8 frames đạt score **0.8255**, thấp hơn rõ rệt so với mức **0.8510** của cấu hình chuẩn 16 frames.
- **Về tài nguyên:** Tốc độ huấn luyện khi giảm xuống 8 frames nhanh hơn xấp xỉ **2 lần**, giúp tiết kiệm đáng kể thời gian tính toán và dung lượng VRAM.

**Phân tích và Kết luận:** Sự sụt giảm độ chênh lệch điểm số (Acc) cho thấy tầm quan trọng của việc duy trì cấu trúc dữ liệu đồng bộ với giai đoạn pretrain. Việc ép số lượng khung hình xuống còn 8 có thể đã gây ra sự sai lệch (mismatch) đối với các **Temporal Positional Embeddings** đã được học sẵn (vốn được tối ưu cho độ phân giải thời gian là 16). Khi các tham số nhúng vị trí không còn tương thích hoàn toàn với dữ liệu đầu vào, khả năng nắm bắt chuyển động của mô hình bị suy giảm, dẫn đến kết quả kém hơn dù tốc độ xử lý tăng lên. Do đó, để đạt được độ chuẩn xác cao nhất, việc giữ nguyên 16 frames vẫn là lựa chọn tối ưu nhất cho VideoMAE.