# Tài liệu ôn tập Toán ứng dụng & Cơ sở Học máy (Bộ dữ liệu đánh giá Retrieval)

Giải phương trình bậc nhất 3x + 5 = 26. Trừ 5 ở hai vế rồi chia cho 3, ta được x = 7.

Hình chữ nhật có chu vi 36 cm và chiều dài 11 cm. Vì 2 × (chiều dài + chiều rộng) = 36, chiều rộng là 7 cm và diện tích là 77 cm².

Dãy số cộng là 2, 5, 8, 11, ... Công sai bằng 3, nên số hạng thứ 10 là 2 + 9 × 3 = 29.

Một mảnh vườn hình chữ nhật khác có chu vi 48 m và chiều dài 15 m. Áp dụng công thức nửa chu vi 24 m, chiều rộng mảnh vườn bằng 9 m và diện tích đạt 135 m².

Cấp số nhân có số hạng đầu u1 = 3 và công bội q = 2. Số hạng thứ 6 của cấp số nhân được tính theo u1 × q^5 = 3 × 32 = 96, còn tổng 6 số hạng đầu bằng 189.

Phương trình bậc hai x² - 7x + 10 = 0 có biệt thức Δ = 49 - 40 = 9 > 0. Theo công thức nghiệm và định lý Viète, hai nghiệm phân biệt là x1 = 2 và x2 = 5, tổng hai nghiệm bằng 7.

Trong xét nghiệm y tế, tỷ lệ mắc bệnh nền trong cộng đồng là 1% (prior = 0.01). Xét nghiệm có độ nhạy (sensitivity) 99% và tỷ lệ dương tính giả 5%. Theo định lý Bayes, xác suất thực sự mắc bệnh khi có kết quả dương tính chỉ xấp xỉ 16.6%.

Phân rã giá trị kỳ dị (SVD) tách ma trận thực A kích thước m × n thành U Σ V^T. Trong thuật toán giảm chiều PCA, để giữ lại tối thiểu 95% phương sai tích lũy trên tập dữ liệu ảnh chữ viết tay, thực nghiệm chọn k = 42 thành phần chính đầu tiên.

Số điều kiện của ma trận khả nghịch A được định nghĩa là κ(A) = σ_max / σ_min. Khi κ(A) = 10^8, hệ phương trình tuyến tính Ax = b trở nên kém ổn định số học (ill-conditioned), nhiễu nhỏ ở vế phải gây sai số lớn ở nghiệm.

Với hàm lồi khả vi có gradient thỏa điều kiện Lipschitz với hằng số L = 4, thuật toán Gradient Descent hội tụ đơn điệu khi bước nhảy cố định thỏa mãn 0 < η < 2/L, trong đó bước nhảy tối ưu lý thuyết thường đặt η = 1/L = 0.25.

Trong huấn luyện mô hình ngôn ngữ, bộ tối ưu AdamW tách riêng weight decay khỏi cập nhật moment bậc hai. Cấu hình chuẩn sử dụng β1 = 0.9, β2 = 0.999, ε = 1e-8 và hệ số weight_decay = 0.01 kết hợp lịch giảm tốc độ học dạng cosine sau 500 bước warmup.

Thuật toán SGD có quán tính (Momentum) tích lũy vận tốc v_t = μ v_{t-1} + g_t với hệ số μ = 0.9 nhằm giảm dao động theo phương độ cong lớn và tăng tốc độ di chuyển dọc theo các thung lũng hẹp của mặt phẳng hàm mất mát.

Hàm mất mát Cross-Entropy đo độ lệch giữa phân phối dự đoán softmax và nhãn mục tiêu. Khi bật Label Smoothing với hệ số ε = 0.1 cho bài toán phân loại 10 lớp, nhãn đúng nhận xác suất 0.91 thay vì 1.0, giúp giảm hiện tượng quá tự tin (overconfidence).

Phân kỳ Kullback-Leibler D_KL(P || Q) đo lượng thông tin mất đi khi xấp xỉ phân phối thực P bằng phân phối Q. Đại lượng này luôn không âm theo bất đẳng thức Gibbs nhưng không đối xứng, nên không phải là một metric khoảng cách thực sự.

Scaled Dot-Product Attention computes softmax(Q K^T / sqrt(d_k)) V. Dividing the dot products by sqrt(d_k) prevents large inner products in high dimensions from pushing the softmax function into regions with vanishingly small gradients.

Low-Rank Adaptation (LoRA) freezes pretrained Transformer weights and injects trainable rank-decomposition matrices A and B into attention projections. Setting rank r = 8 and scaling alpha = 16 reduces trainable parameters by over 99% while preserving downstream accuracy.

Bi-encoders encode queries and passages independently into fixed-dimensional vectors, enabling sub-linear approximate nearest neighbor search via HNSW. Cross-encoders concatenate query and passage tokens for full self-attention across all layers, achieving higher ranking precision at linear inference cost per candidate.

Standard self-attention materializes an N × N attention matrix in GPU high-bandwidth memory (HBM), incurring quadratic memory traffic. Tiling blocks into fast on-chip SRAM avoids writing intermediate score matrices to HBM and yields exact attention with linear memory footprint.

Extractive grounding mitigates extrinsic hallucinations in Retrieval-Augmented Generation by instructing the language model to abstain or explicitly flag unsupported claims whenever retrieved passages lack verifiable evidence for the user query.

Thuật toán Reciprocal Rank Fusion (RRF) hợp nhất nhiều danh sách xếp hạng mà không cần chuẩn hóa thang điểm giữa dense vector và BM25. Công thức cộng dồn 1 / (k_rrf + rank) thường chọn hằng số làm mượt k_rrf = 60 để giảm ảnh hưởng cực đoan của các ngoại lệ đứng hạng 1.

Khi chia nhỏ văn bản PDF phục vụ tìm kiếm ngữ nghĩa, kích thước đoạn chunk_size = 500 ký tự kèm phần gối nhau chunk_overlap = 50 ký tự giúp giữ nguyên ngữ cảnh các câu nằm vắt qua ranh giới hai đoạn liên tiếp.

Trong mạng nơ-ron sâu trên tập dữ liệu nhỏ, đặt tỷ lệ Dropout p = 0.3 ở các tầng ẩn kết hợp dừng sớm (Early Stopping) với độ kiên nhẫn patience = 7 epoch trên tập validation giúp ngăn chặn overfitting hiệu quả.

Trên dữ liệu mất cân bằng nghiêm trọng (tỷ lệ lớp hiếm chỉ 0.5%), độ chính xác tổng thể Accuracy dễ gây hiểu lầm ở mức 99.5%. Khi đó cần đánh giá bằng F1-score hoặc diện tích dưới đường cong Precision-Recall (PR-AUC); còn với bài toán xếp hạng tài liệu thì dùng MRR và nDCG.

Lượng tử hóa trọng số mô hình từ FP16 xuống định dạng 4-bit NormalFloat (NF4) giảm gần 75% dung lượng VRAM cần thiết khi suy luận, chỉ làm tăng perplexity không đáng kể nếu giữ các kênh ngoại lai nhạy cảm ở độ chính xác 16-bit.
