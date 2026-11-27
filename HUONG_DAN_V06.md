MintoAI V0.6 — học quy tắc có kiểm chứng và hợp nhất hai checkpoint

Mục tiêu của bản thử này
Mạng dùng khi dự đoán là một UnifiedRig. Hai checkpoint cũ được dùng làm
teacher trong lúc học. Rig làm sẵn là chuẩn đánh giá chính. Kết quả sinh
từ PSD2Live bị tắt trong cấu hình mặc định.

Cài đặt
1. Ctrl+C trong hai trainer cũ, đợi thông báo Saved, rồi đóng dashboard cũ.
2. Giải nén gói này vào C:\MintoAI.
3. Giữ snapshot, models, hai checkpoint cũ và môi trường .venv đang có.
   Các file process_rig_dataset.py và Cubism SDK hiện có được dùng lại
   nếu bạn chạy công cụ tạo đề PSD2Live tùy chọn.

Lượt thử đầu tiên
Tại C:\MintoAI, dùng một cửa sổ PowerShell cho trainer mới.

```powershell
.\.venv\Scripts\python.exe .\training\train_unified_rig.py --psd2live C:\PSD2Live --device cuda --pose-batch auto --steps 2000 --eval-every 200 --confirm-authorized
```

Một cửa sổ khác cho dashboard.

```powershell
.\.venv\Scripts\python.exe .\minto_dashboard_v5.py
```

Lượt 2.000 bước dùng để kiểm tra vòng thực nghiệm và xem sai số có cải thiện.
Để tiếp tục không giới hạn từ checkpoint mới, chạy lệnh sau.

```powershell
.\.venv\Scripts\python.exe .\training\train_unified_rig.py --device cuda --resume --confirm-authorized
```

Resume tự khôi phục chu kỳ kiểm tra và chương trình học đã lưu.
Nếu bạn chỉ muốn chạy thêm một đoạn, thêm --steps cùng số bước bổ sung.

Đầu vào mặc định
- checkpoints/real_rig_curriculum.pt
- checkpoints/fullbody_extreme_v04.pt

Bạn có thể chỉ định tên khác bằng --real-checkpoint và --extreme-checkpoint.
Snapshot và models được lấy từ cấu hình trong checkpoint Real Rig.
Nếu teacher cũ thiếu cấu hình snapshot, cung cấp --data và --models.
Dữ liệu và teacher dùng cho một lượt phải giữ nguyên khi resume.

Kiến thức được tận dụng thế nào
- Encoder CNN, các khối đồ thị và đầu dự đoán tương thích được khởi tạo
  từ Real Rig. Bộ điều khiển được mở rộng sang 13 ID parameter.
- Extreme teacher được nạp với đúng thứ tự 30 đặc trưng đầu vào cũ.
  Nó hướng dẫn mạng chung trên hình học tổng hợp có bộ phận và mesh.
- Công thức hình học tổng hợp vẫn cung cấp nhãn chính xác cho bài xoay.
  Teacher chỉ được bắt chước ở các điểm nó đang chính xác hơn student.
  Khi student dự đoán tốt hơn teacher, loss bắt chước tại điểm đó bằng 0.
- Những ví dụ tập bị sai cao được lưu vào replay để ôn nhiều hơn.
  Các pose thi không được đưa vào replay.
- Nhân vật của validation/test và một số biến thể cùng tên được giữ riêng
  trong lượt mới. Nếu checkpoint cũ đã từng học một biến thể đó thì lịch sử
  tiếp xúc này không thể được xóa bằng cách chia tập mới.

Học quy tắc PSD2Live
Source checkout được đọc tại bốn file đã biết ở commit được ghi trong
*.algorithm_curriculum.json
- ThinPlateSpline.kt
- FaceTurn.kt
- BodyMotionTuning.kt
- EyeLegacyMotion.kt

Bộ đọc ghi lại hash file, tên hàm và các giá trị khởi tạo liên quan.
Các thành phần có thể học hiện gồm
- Phép biến đổi affine theo mesh.
- Xoay với góc và tâm xoay do mạng dự đoán.
- Thin-plate spline với độ dịch các điểm điều khiển và regularization học được.
- Kết hợp các quy tắc với phần dự đoán residual của mạng.

Đây là thư viện quy tắc được tham số hóa, dùng source PSD2Live làm tài liệu
và một số giá trị khởi tạo. AI tự chọn cách thử và điều chỉnh tham số bằng
gradient. Gói này chưa tích hợp mô hình ngôn ngữ tự hiểu mọi mã Kotlin,
tự viết thuật toán bất kỳ hoặc tự sửa repository PSD2Live.
Thuật toán FaceTurn đầy đủ, việc dò landmark, opacity, clipping, vật lý
và xuất project CMO3 chưa được chuyển thành bộ học trong bản này.

Vòng thực nghiệm
1. Chọn một cách tính loss hoặc một họ quy tắc.
2. Tập một đoạn trên các model và replay.
3. Chấm trên cùng bộ đề validation.
4. Chỉ nhận mốc mới khi chất lượng rig thật cải thiện, lỗi tọa độ lớn nhất
   của rig thật không tăng và các tiêu chí bảo vệ khác vẫn đạt.
5. Nếu kém đi, khôi phục trọng số và optimizer của mốc tốt nhất.
6. Ghi lại bối cảnh, phương án, điểm trước/sau và quyết định vào SQLite
   và JSONL. Các phương án có kết quả xấu bị tạm hạn chế thử lại.
7. Khi lỗi tập trung vào vùng đứng yên hoặc vùng chuyển động, thống kê
   lựa chọn được ghi riêng cho bối cảnh đó.

Không lấy giảm lỗi trên dữ liệu tổng hợp để bù cho tăng lỗi trên rig thật.
Lượt fixed dùng bộ bài tổng hợp cố định. Khi đủ ba lần kiểm tra đạt ngưỡng,
chuyển sang sinh tỷ lệ/pose mới; qua lượt đó thì trở lại bộ fixed để ôn.
Ngưỡng giữ nguyên, mặc định là 1e-9.
Chương trình học trên tất cả nguồn từ đầu; một model chưa PASS không chặn
mọi model khác như trainer tuần tự trước đây.

PSD2Live sinh model sai thì xử lý thế nào
Cấu hình mặc định chỉ dùng thư viện quy tắc đọc từ source. Để thử thêm
kết quả sinh từ PSD2Live, bạn chủ động tạo đề và truyền --psd-index.

```powershell
.\.venv\Scripts\python.exe .\extractor\prepare_psd2live_teacher.py --psd2live C:\PSD2Live --psd-dir .\models --confirm-authorized
```

Công cụ gọi build và CLI của checkout PSD2Live, xuất model vào dataset,
rồi trích pose qua Cubism Core hiện có. Mỗi sweep khởi tạo MOC mới để
tránh trạng thái từ sweep trước ảnh hưởng pose gốc.
Cần JDK/build dependencies cho checkout, Playwright/Edge và SDK đang có.
Windows/JDK/Cubism export chưa được chạy thử trong môi trường kiểm tra này.

Sau đó, dùng một output mới và thêm
--psd-index .\dataset\psd2live_teacher\index.json

Các nhãn này chỉ có mặt trong một loại thử nghiệm psd_hint, với trọng số 0,02.
Validation rig thật quyết định có nhận cải tiến hay không. Trùng với
PSD2Live không được tính là rig đúng. Các nguồn và hash được lưu vào checkpoint.

Tài nguyên
Trainer mới chạy CNN/GNN và các rule trên CUDA khi dùng --device cuda.
Nó vector hóa nhiều pose của cùng graph, thay vì forward từng pose.
Auto batch benchmark đo poses/giây và peak VRAM trên graph lớn,
giữ khoảng 20% VRAM dự phòng và chọn cấu hình đo được nhanh nhất.
Batch được giảm khi forward/backward báo CUDA OOM.
Graph inputs được cache theo số lượng, mặc định 4, để giảm truyền CPU/GPU.
--cache-graphs đổi giới hạn này. --max-pose-batch đổi giới hạn benchmark.
Không có mục tiêu ép đầy VRAM; số đo tốc độ quyết định batch được chọn.

Float64 là mặc định để theo đuổi ngưỡng số học thấp. Nó có thể chậm hơn
trên GPU laptop. Dữ liệu rig thật đã đi qua float32, nên float64 không
phục hồi được thông tin đã làm tròn. 1e-9 là ngưỡng kiểm tra trên tọa độ
đã chuẩn hóa, chưa phải chứng nhận render giống hệt.
Gói này chạy thủ công. Dashboard theo dõi nhiệt, không điều khiển trainer.
Chưa benchmark trên RTX 3050/Windows của bạn.

File được tạo
- unified_rig_v06.pt — trạng thái mới nhất để resume.
- unified_rig_v06.best.pt — mốc được chấp nhận tốt nhất để dự đoán.
- unified_rig_v06.metrics.jsonl — điểm kiểm tra và trạng thái dashboard.
- unified_rig_v06.lessons.jsonl — giả thuyết, bằng chứng, quyết định.
- unified_rig_v06.experience.sqlite — lịch sử thực nghiệm.
- unified_rig_v06.algorithm_curriculum.json — tài liệu/quy tắc source đã đọc.

Số bước của trainer chung bắt đầu từ 0. Trọng số Real Rig và kiến thức
Extreme được tái sử dụng; số bước của hai teacher cũng được ghi trong sổ.
Các checkpoint cũ không bị ghi đè.

Xem thử mạng chung sau khi train
```powershell
.\.venv\Scripts\python.exe .\training\view_unified_rig.py --data DUONG_DAN_SNAPSHOT --model zundamon
```

HTML được lưu tại dataset/unified_comparison.html. Đây là so sánh lưới mesh,
có dùng texture làm đầu vào CNN nhưng chưa render texture/clipping/opacity.
Lúc dự đoán chỉ cần checkpoint mạng chung và mesh/texture đầu vào.
Hai checkpoint teacher chỉ cần cho lúc train.

Kiểm tra đã thực hiện trên CPU và dữ liệu thử
- Nạp kiến thức encoder Real Rig và đối chiếu phép tính teacher dạng batch.
- Kiểm tra thứ tự đặc trưng Extreme, chín control, hệ tọa độ và scale.
- Cả nguồn rig thật, Extreme và đề thử PSD đều cập nhật cùng một student.
- Teacher tệ hơn student không thêm loss bắt chước.
- Cải thiện lớn trên tổng hợp không thể che việc rig thật kém đi.
- TPS thỏa ràng buộc điểm, tái tạo affine và có gradient.
- Phép xoay khớp công thức với sai số float64.
- Lưu/resume, vòng fixed/random/review và checkpoint dự đoán độc lập.
- Dashboard đọc checkpoint mới, log và đổi ngôn ngữ trong kiểm tra DOM.

Chưa có kết quả chứng minh bản này vượt model hiện tại của bạn.
Bản này dùng bộ quy tắc có kiểm chứng theo lựa chọn của bạn. Agent ngôn ngữ là phần mở rộng tùy chọn sau này.
