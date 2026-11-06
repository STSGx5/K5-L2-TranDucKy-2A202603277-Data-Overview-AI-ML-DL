# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
+ Record này nói về toàn bộ bức ảnh.
+ Lý do: Trong mô hình phân loại ảnh tiêu chuẩn, nhãn dự đoán mô tả ngữ cảnh hoặc chủ thể chính chi phối toàn bộ khung hình ảnh, chứ không cắt khoanh vùng từng đối tượng riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
+ Danh sách tên lớp đi kèm theo taxonomy/checkpoint của mô hình. Mô hình không tự "nghĩ" ra tên lớp mới mà chỉ chọn ra xác suất tương ứng với các lớp đã được định nghĩa sẵn trong tập nhãn của ImageNet.
5. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
+ class_id: Giúp hệ thống máy tính, cơ sở dữ liệu hoặc code xử lý, so khớp, tính toán tự động một cách chính xác, tránh các lỗi do sai chính tả hoặc khác biệt ngôn ngữ chữ viết.
+ class_name: Giúp con người đọc, kiểm tra và hiểu trực quan ngay lập tức nhãn đó là gì mà không cần tra cứu bảng mã số.
+ taxonomy_name: Định nghĩa rõ tập từ vựng và bộ quy chuẩn nào đang được sử dụng. Cùng một tên lớp cab có thể có ý nghĩa hoặc ID khác nhau ở các bộ taxonomy khác nhau; do đó cần có taxonomy_name để định hướng ngữ cảnh chuẩn xác.
6. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
+ Chủ thể chính: Quy định rõ có lấy chủ thể chiếm diện tích lớn nhất, chiếm trung tâm bức ảnh, hay chủ thể đóng vai trò cốt lõi của ngữ cảnh hay không.
+ Quy tắc đa nhãn: Xác định rõ ảnh chỉ được gán một nhãn duy nhất hay được phép gán nhiều nhãn nếu trong ảnh xuất hiện nhiều chủ đề song song.
+ Mức độ cụ thể: Quy định gán nhãn ở mức độ chung chung hay mức độ chi tiết cụ thể.
7. Vì sao không được lấy `score` của model làm ground truth?
+ Score chỉ là độ tự tin: Điểm số chỉ phản ánh mức độ chắc chắn của mô hình AI đối với dự đoán của nó dựa trên các đặc trưng học được, chứ không đảm bảo đó là sự thật khách quan tuyệt đối.
+ Ground Truth đòi hỏi tính chính xác tuyệt đối: Ground truth phải đến từ con người gán nhãn có kiểm định hoặc từ dữ liệu thực tế chính xác, nhằm phục vụ cho việc đánh giá, huấn luyện khách quan, tránh việc "lấy kết quả của máy để chứng minh máy đúng".

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
+ Vị trí góc trên-trái: Nằm tại tọa độ pixel (385.33, 69.24)– cách mép trái ảnh khoảng 385.33 pixel và cách mép trên ảnh khoảng 69.24 pixel.
+ Vị trí góc dưới-phải: Nằm tại tọa độ pixel (498.92, 348.92) – cách mép trái ảnh khoảng 498.92 pixel và cách mép trên ảnh khoảng 348.92 pixel.
+ Kích thước: Hộp giới hạn có chiều rộng bằng 113.58 pixel và chiều cao bằng 279.68 pixel, bao trọn khu vực thân thể của đối tượng person trong ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
  + Ngưỡng cao (0.60): 6 prediction 
  + Ngưỡng thấp (0.20): 11 predictions
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
+ Loại bỏ nhiễu: Kiểm tra và xóa bỏ các hộp dự đoán sai lệch do mô hình nhận diện nhầm nền hoặc vật thể không chính xác.
+ Xử lý chồng lấn: Lọc bỏ các hộp bị trùng lặp hoặc điều chỉnh lại biên độ của hộp cho khít với vật.
+ Tăng thời gian kiểm tra thủ công: Tốn nhiều công sức hơn để duyệt qua danh sách dài các đối tượng có độ chắc chắn thấp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
+ Bao trọn vẹn: Hộp giới hạn phải ôm trọn toàn bộ phần hiển thị rõ ràng của đối tượng, không để lòi các bộ phận chính ra ngoài biên hộp.
+ Bám sát biên vật thể: Cạnh của hộp phải tiếp xúc sát với biên ngoài cùng của đối tượng, tránh để khoảng trống dư thừa quá lớn làm lãng phí không gian tính toán.
+ Vật thể bị che khuất: Nếu đối tượng bị che một phần bởi vật khác, chỉ vẽ box bao phủ phần thấy được của đối tượng đó theo đúng ranh giới hiển thị, không tự ý đoán mò phần bị che khuất trừ khi quy định riêng của dự án yêu cầu ước lượng toàn bộ.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
+ hộp dự đoán bowl 0.38 bị cắt mép ngay tại biên cạnh trái của khung hình, khiến một phần của chiếc bát bị tràn ra ngoài giới hạn ảnh và không hiển thị trọn vẹn
+ Đưa vào viết thành quy tắc trong guideline
+ Lý do: Việc vật thể bị cắt mép bởi khung hình là một hiện tượng phổ biến và chuẩn hóa trong các bài toán gán nhãn nhận diện đối tượng. Guideline chuẩn chỉ cần quy định rõ ràng cách xử lý, giúp người gán nhãn tự chủ động thực hiện đồng nhất cho tất cả các ảnh tương tự mà không cần hỏi lại quản l
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record:
+ `instance_id`=kitchen-002 ;
+ `class_name`=bowl ;
+ `score`=0.735743 ;
+ số điểm polygon=`polygon_point_count`=67 ;
+ 3–5 điểm đầu của `polygon_xy`=[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
+ Độ chính xác hình học chi tiết: Bounding box chỉ là một hình chữ nhật bao ngoài, chứa rất nhiều khoảng trống hoặc lẫn các vật thể khác. Trong khi đó, polygon bám sát theo đúng đường cong, mép cạnh thực tế của vật thể.
+ Tách bạch cá thể rõ ràng: Giúp mô tả chính xác diện tích, hình thù cụ thể của từng đối tượng riêng biệt, cực kỳ hữu ích cho các bài toán đòi hỏi bóc tách pixel.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
+ Phân biệt định danh từng cá thể: instance_id dùng để phân biệt rõ ràng giữa các đối tượng cùng một lớp trong ảnh.
+ Không phải class ID: class ID cho biết vật thể thuộc loại gì, còn instance_id cho biết đây là cá thể cụ thể nào trong số các vật cùng loại đó.
+ Không phải tracking ID: tracking ID dùng để đồng bộ và theo dõi một đối tượng xuyên suốt qua nhiều khung hình. Trong khi instance_id ở đây chỉ mang tính định danh tĩnh trong một bức ảnh tĩnh đơn lẻ.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
+ Bám sát biên: Đường viền đa giác phải ôm sát từng điểm chuyển đổi pixel giữa vật thể và nền. Không được để lộ khoảng trống lớn hoặc lấn chiếm sang nền xung quanh.
+ Độ phân giải điểm: Đặt mật độ điểm polygon vừa đủ dày ở các đoạn cong để tạo đường cong mượt mà, và thưa hơn ở các đoạn thẳng để tối ưu hóa dữ liệu.
+ Xử lý vùng biên: Tuyệt đối không vẽ tràn vào vật thể khác che khuất nó; nếu có hiện tượng chồng lấn, đường viền của vật nằm phía trước phải đè lên vật nằm phía sau theo đúng lớp hiển thị.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
+ Trên bề mặt của chiếc bàn, có rất nhiều đồ vật đặt sát hoặc chồng lên nhau, các chiếc bát đặt ngay trên bề mặt bàn
+ Sử dụng Guideline
+ Tình trạng các vật thể đặt sát, chạm mặt hoặc chồng lấn lên nhau là hiện tượng rất phổ biến trong các bộ dữ liệu phân đoạn thực tế. Các quy tắc phân tách lớp, thứ tự che khuất, và cách vẽ ranh giới khi các vật chạm nhau đều đã được chuẩn hóa trong tài liệu hướng dẫn gán nhãn của dự án để người gán nhãn tuân thủ đồng bộ
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn dán toàn cục cho cả bức ảnh | Mô hình dự đoán nhầm ngữ cảnh hoặc ảnh có nhiều chủ đề nhưng chỉ lấy một nhãn. | Đọc guideline, chọn một nhãn chính xác nhất đại diện cho toàn bộ khung hình ảnh. | Kiểm tra xem nhãn được chọn có phản ánh đúng chủ thể/ngữ cảnh chi phối toàn bộ bức ảnh hay không. |
| Phát hiện vật thể | Hộp giới hạn định dạng tọa độ kèm theo class_id và class_name. | Các vật thể bị cắt mép khung hình hoặc vật thể bị che khuất một phần. | Vẽ hộp giới hạn ôm trọn vẹn phần hiển thị của vật thể, tuân thủ quy tắc bám sát mép biên ngoài. | Kiểm tra độ khít của hộp đảm bảo không để khoảng trống thừa quá lớn và các vật bị cắt mép được xử lý đúng chuẩn. |
| Instance segmentation | Vùng mặt nạ nhị phân chi tiết bám theo đường viền kèm instance_id phân biệt cá thể. | Các vật thể đặt sát hoặc chạm nhau trên bề mặt biên mờ hoặc chồng lấn lớp | Chấm các điểm polygon bám sát theo đường cong và đường viền thực tế của từng cá thể riêng biệt. | Kiểm tra độ mượt và chính xác của đường viền đa giác, đảm bảo các vật chạm nhau không bị lấn chiếm hoặc đè nhầm lớp. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
+ Luôn bảo mật thông tin, không tiết lộ hoặc chia sẻ dữ liệu nhãn và hình ảnh của bài lab ra bên ngoài phạm vi học tập và công việc được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
+ Nên dừng lại và báo cáo trực tiếp cho Lab Coach hoặc người quản lý phụ trách hướng dẫn lớp lab.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [X] Đã nộp **link repository** trên LMS
