# Proposal: Cải tiến segmentation & đánh giá thao tác công nhân bằng VLM

**Dự án:** `poc-ai-eval` — đối chiếu video công nhân (worker) với video chuyên gia (expert) để đánh giá thực hành công đoạn may.
**Ngày:** 2026-07-03
**Trạng thái:** Đã triển khai code đầy đủ (P1–P4, xem README). Chạy thực tế
tầng VLM cần `OPENROUTER_API_KEY`; P0 (chạy lại extraction + ROI trên server
RTX 5060) cần quyền truy cập server.

---

## 1. Bối cảnh & vấn đề

Pipeline hiện tại (xem `plan.md`, `README.md`):

```
expert.mp4 + course JSON ──► per-frame signals (WiLoR hand keypoints,
worker.mp4               ──►  optical flow quanh tay, DINOv2 crop)
        ──► similarity matrix (DTW keypoint/flow + duration + Chamfer NN + image_embed)
        ──► Viterbi decode theo thứ tự scene expert
        ──► alignment report
```

Kết quả hiện tại **sai về bản chất**: report luôn ra "17/17 matched, 0 lỗi" nhưng ranh giới segment không khớp với thao tác thực tế của công nhân (ví dụ: `Lại mũi bằng cần gạt` — expert 2.0s — bị gán 22.4s của worker).

### Chẩn đoán (đo trên `outputs/` ngày 03/07)

| # | Phát hiện | Bằng chứng | Hệ quả |
|---|-----------|------------|--------|
| 1 | Ma trận similarity không phân biệt được scene | Điểm dồn cục 0.63 ± 0.065; chênh lệch best-vs-mean mỗi step chỉ ~0.11; argmax per-step nhảy loạn (1→4→11→13→2→5…) | Emission ≈ nhiễu |
| 2 | Viterbi "vẽ" đường đi giả | Penalty backward=4.0, reenter=4.0, skip=1.5 → khi emission nhiễu, đường rẻ nhất luôn là đi tuần tự đủ 17 scene | Report luôn "0 lỗi" bất kể công nhân làm gì; ranh giới do dwell-prior quyết định, không phải nội dung video |
| 3 | Hand detection fail nặng trên video worker | Video 480×368: tay trái mất 53% frame, tay phải 37%; NaN được nội suy đường thẳng | Hơn nửa quỹ đạo keypoint đưa vào DTW là dữ liệu bịa |
| 4 | Term thị giác mạnh nhất đang tắt | `embeddings.npy` không tồn tại trong `outputs/` (extraction chạy version cũ) → `image_embed` (weight 0.30) bị bỏ qua | Chỉ còn pose/flow vốn đã nhiễu |

**Kết luận:** kiến trúc (similarity + Viterbi ràng buộc thứ tự) hợp lý, nhưng đứng trên nền feature không mang thông tin. Cần thay nguồn tín hiệu emission, không cần đập lại toàn bộ.

---

## 2. Ràng buộc thiết kế (từ yêu cầu nghiệp vụ)

1. **Scene composite:** một scene expert có thể chứa ≥2 thao tác — thường 1 thao tác chính + thao tác phụ; **thao tác phụ có thể sub-second (vài trăm ms)**. Không thể segment thao tác phụ bằng sampling thưa — phải kiểm tra *sự hiện diện* của nó bên trong scene đã căn.
2. **Chỉ đánh giá vùng thao tác tay** (khu vực kim / chân vịt / hai tay khi may). Phần còn lại của khung hình là nhiễu, không đưa vào cả feature lẫn VLM.
3. POC không train/fine-tune model. VLM gọi qua **OpenRouter** (API key sẽ được cung cấp). Server **RTX 5060** dùng cho extraction (WiLoR/DINOv2 CUDA).
4. Chi phí API phải kiểm soát được: cache kết quả VLM theo (video hash, frame, prompt version) để chạy lại không tốn phí.

---

## 3. Thiết kế đề xuất: 2 tầng

### Tầng 1 — Căn scene (thao tác chính) bằng VLM emission + Viterbi

Giữ nguyên khung decode, thay nguồn emission:

```
worker.mp4
  ├─► work-area ROI cố định (mục 3.3) ──► crop + upscale
  ├─► change-point (đã có) ──► cắt thô thành ~15–25 đoạn ứng viên
  ├─► mỗi đoạn: sample 3–5 keyframe (crop) ──► VLM classify
  │     prompt = danh sách 17 scene (tên thao tác VN + mô tả ngắn
  │              + 1–2 frame tham chiếu crop từ expert video làm few-shot)
  │     output = phân bố điểm / top-k scene + confidence (JSON)
  ├─► emission(step, scene) = VLM score, trộn với motion/flow hiện có
  └─► Viterbi decode (giữ nguyên, tinh chỉnh penalty — mục 3.4)
```

Chi tiết:

- **Sampling:** ưu tiên "change-point trước, classify sau" thay vì chấm 328 step × 17 scene. Video 65s ⇒ ~20 đoạn × 4 frame ≈ 80 ảnh crop, gom batch nhiều frame/request. Ước tính < 30 call/video với Gemini Flash hoặc Qwen2.5-VL qua OpenRouter — chi phí không đáng kể.
- **Prompt:** liệt kê scene theo thứ tự, kèm mô tả hành vi quan sát được ("hai tay đặt nẹp vào dưới chân vịt", "tay phải gạt cần lại mũi"…) sinh từ tên operation + xác nhận thủ công 1 lần. Few-shot bằng frame expert đã crop cùng ROI.
- **Output hợp đồng:** JSON `{segment_id, scores: {scene_index: 0..1}, evidence: "..."}` — giữ trường evidence để debug.
- **Fallback:** VLM score là term mới trong `WEIGHTS` (trọng số lớn, ~0.6); các term pose/flow giữ vai trò phụ và là fallback khi không có API key.

### Tầng 2 — Kiểm tra thao tác phụ trong từng segment đã căn

Sau khi tầng 1 gán đoạn worker ↔ scene expert:

- Với mỗi scene có thao tác phụ (theo course JSON), hỏi câu **yes/no có định hướng**: "trong đoạn này, công nhân có [gạt cần lại mũi / nhấn nút / cắt chỉ]… không?"
- **Frame dày quanh đỉnh motion:** trong segment, lấy các cụm 5–8 frame liên tiếp ở full fps quanh đỉnh optical-flow/change-point (thao tác phụ ngắn luôn tạo motion spike) → đưa cụm frame cho VLM.
- Thao tác phụ có chữ ký vật lý rõ (nhấn nút, gạt cần — vị trí cố định trên máy) có thể thêm detector flow cục bộ tại vùng nút/cần làm tín hiệu bổ trợ, rẻ hơn VLM.
- **Không cố xác định timestamp chính xác của thao tác phụ** trong POC — chỉ kết luận present / absent / uncertain.

### 3.3 Work-area ROI cố định (điều kiện tiên quyết cho cả 2 tầng)

- Xác định 1 lần per video: union các hand bbox đã detect được qua toàn video + vị trí kim (`roi_auto.json`), hoặc cho người dùng vẽ 1 lần per camera setup.
- **Crop + upscale ROI trước khi chạy WiLoR** → kỳ vọng giảm mạnh tỉ lệ mất detection (hiện 37–53%) vì tay chiếm nhiều pixel hơn.
- Mọi frame gửi VLM đều là crop ROI này — ít nhiễu nền, chi tiết tay–vải–kim rõ hơn, rẻ hơn per call.

### 3.4 Sửa hành vi report "0 lỗi giả tạo" (không cần VLM, làm ngay)

- Thêm ngưỡng **no-confident-match**: segment có emission dưới ngưỡng → `LOW_CONFIDENCE` / `UNMATCHED` thay vì ép match.
- Cân lại penalty Viterbi khi emission đã có nghĩa (backward/reenter hiện quá cao so với thang emission mới).
- Report thành 2 phần: (a) scene-level matched/missing/wrong-order/timing như cũ; (b) **checklist thao tác phụ** present/absent/uncertain per scene — không ép thao tác phụ thành segment tuyến tính.

---

## 4. Kế hoạch triển khai

| Phase | Nội dung | Phụ thuộc | Ước lượng |
|-------|----------|-----------|-----------|
| **P0** | Chạy lại extraction bằng code hiện tại trên server 5060 (sinh `embeddings.npy`, bật lại `image_embed`) — baseline mới | GPU server | 0.5 ngày |
| **P1** | Work-area ROI cố định + crop/upscale trước WiLoR; đo lại tỉ lệ hand detection | P0 | 1 ngày |
| **P2** | Module VLM emission tầng 1: sampling, prompt, OpenRouter client, cache, trộn emission, tinh chỉnh Viterbi + no-confident-match | API key, P1 | 2–3 ngày |
| **P3** | Tầng 2: checklist thao tác phụ (motion-spike sampling + VLM yes/no + detector nút/cần) | P2 | 1–2 ngày |
| **P4** | Report v2 (scene alignment + aux checklist), cập nhật timeline HTML, đánh giá trên video mẫu | P2, P3 | 1 ngày |

**Tiêu chí nghiệm thu:**
- P1: tỉ lệ frame mất hand detection trên worker video giảm còn < 15%.
- P2: ranh giới segment khớp mắt thường khi soi cùng video (đặc biệt cụm `Diễu cạnh dài` 4 lần lặp); report dám báo `LOW_CONFIDENCE`/`MISSING` khi cắt thử scene khỏi video.
- P3: phát hiện đúng present/absent thao tác phụ trên các case thử (che/cắt đoạn nhấn nút…).
- P4: người không kỹ thuật đọc report hiểu được công nhân sai gì, ở giây thứ mấy.

---

## 5. Rủi ro & phương án

| Rủi ro | Phương án |
|--------|-----------|
| VLM nhầm giữa các scene giống nhau (4 lần `Điều chỉnh mép + Diễu cạnh dài` liên tiếp) | Chấp nhận ở tầng emission — Viterbi + thứ tự + duration prior sẽ phân giải; các scene lặp cùng nhãn chỉ cần đúng *nhóm* |
| Thao tác phụ quá ngắn, không có motion spike rõ | Đánh dấu `UNCERTAIN` thay vì kết luận sai; ghi nhận làm hạn chế POC |
| Chi phí/độ trễ API khi scale nhiều video | Cache theo (video hash, frame, prompt version); batch frame; chọn model rẻ (Gemini Flash / Qwen2.5-VL) |
| Video worker chất lượng thấp hơn nữa (mờ, rung) | ROI crop + upscale giảm thiểu; nếu vẫn kém → yêu cầu chuẩn quay tối thiểu (đưa vào hướng dẫn nghiệp vụ) |
| Không có ground truth để đo chính xác | Gán nhãn tay 1–2 video worker (chỉ ranh giới scene) làm bộ đo cho P2/P4 |

---

## 6. Ngoài phạm vi POC này

- Train/fine-tune model nhận diện thao tác.
- Xác định timestamp chính xác của thao tác phụ sub-second.
- Real-time / streaming; multi-camera.
- Tự động sinh mô tả scene hoàn toàn không cần xác nhận thủ công.

---

## 7. Việc cần từ phía anh/chị

1. **OpenRouter API key** (chặn P2).
2. Xác nhận quyền truy cập server RTX 5060 và cách deploy (SSH? đường dẫn dữ liệu?).
3. (Khuyến nghị) 1–2 video worker kèm nhận xét của chuyên gia về lỗi thực tế — làm ground truth đánh giá.
