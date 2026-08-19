# Báo cáo nghiệm thu Lab 19

**Sinh viên:** Phạm Gia Bảo  
**MSSV:** 2A202601506  
**Cohort:** A20-K4  
**Môi trường ghi nhận trong notebook:** Lite, Qdrant in-memory, Feast SQLite, CPU

## Căn cứ

Báo cáo này chỉ trích số liệu từ output đã lưu trong các notebook hiện có ở
thư mục gốc. Các notebook `01` đến `08` đều có output. Trạng thái tái lập bên
dưới là kết quả kiểm tra trực tiếp trên workspace ngày 19/08/2026, không suy
diễn từ output cũ.

## Kết quả notebook

| Notebook | Bằng chứng trong output đang lưu |
|---|---|
| NB1 | Corpus 1.000 tài liệu, index 1.000 vectors; truy vấn paraphrase trả 5 kết quả chủ đề `cloud`. |
| NB2 | RRF sensitivity đã chạy; cấu hình depth=50, k=60 đạt 80,8%. BM25 77,8%, Vector 80,4%, Hybrid 80,8%. Exact: Hybrid 98,0%; paraphrase: Vector 43,3%; mixed: Hybrid 97,5%. |
| NB3 | API `/health` trả `ready: True, n_docs: 1000`; Hybrid P50/P95/P99 server-side là 32,4/41,6/46,1 ms, đạt ngưỡng dưới 50 ms. |
| NB4 | Ba Feature View được đăng ký; online lookup P99 5,90 ms; PIT join hiển thị đủ 3 dòng. |
| NB5 | Ở selectivity 3,8%, post-filter recall 0,00 còn Filtered-ANN 1,00; `fetch_k=500` (50% corpus) đạt recall 1,00. |
| NB6 | Budget 16 docs: single-shot recall/balance 0,526/0,08; agentic no-filter 0,906/0,93; agentic + filter 0,823/0,76. Context có Feast features và `doc_ids`. |
| NB7 | Ngưỡng 0,75 có 100% tiết kiệm nhưng 36% false-hit; 0,85 có 100% tiết kiệm và 0% false-hit. Demo cho thấy rò tenant khi không namespace và MISS khi có namespace. |
| NB8 | `session_id`: target-naive gap 0,477, in-fold -0,003. Latest join rò 98,2% dòng; AUC latest/PIT 0,715/0,595; ODFV thay đổi theo amount. |

## Kiểm tra hiện tại

- `make test`: **35 passed, 8 skipped**.
- `make verify-lite`: **chưa đạt** vì thiếu `data/corpus_vn.jsonl`; script yêu cầu chạy `make seed` trước.

Vì vậy, output notebook là bằng chứng đã lưu, nhưng workspace hiện chưa đạt
tính tái lập sạch máy cho đến khi sinh lại dữ liệu và chạy lại chuỗi kiểm tra.
