# Runbook chạy lại và chụp ảnh

## 1. Chuỗi lệnh đã dùng

Tôi chạy các lệnh sau tại thư mục gốc repository trong terminal WSL/Git Bash:

```bash
bash setup-lite.sh
make gen-advanced
make verify-lite
make test
make benchmark
make notebooks
make lab
```

Kết quả hợp lệ phải có `All checks passed`, toàn bộ pytest đạt, benchmark
không lỗi và tám notebook đều in `PASS`. `make lab` mở Jupyter tại
`http://localhost:8888/lab`.

## 2. Danh sách ảnh cần chụp

Mở notebook đã có output, cuộn tới đúng bảng/cell rồi lưu ảnh vào
`submission/screenshots/`:

| Tệp ảnh | Notebook và nội dung cần thấy |
|---|---|
| `nb1_vector_index.png` | NB1: `Indexed: 1000 vectors`, top-5 và paraphrase |
| `nb2_rrf_precision.png` | NB2: bảng Precision@10 và ba slice |
| `nb3_api_p99.png` | NB3: response `/search`, bảng P50/P95/P99 và dòng PASS |
| `nb4_feast_materialize.png` | NB4: ba feature view, materialize, P99 và PIT join |
| `nb5_filtered_ann.png` | NB5: selectivity table và over-fetch ladder |
| `nb6_agentic_retrieval.png` | NB6: ba strategy cùng budget và output context |
| `nb7_semantic_cache.png` | NB7: threshold sweep và tenant isolation demo |
| `nb8_feature_engineering.png` | NB8: leakage table, PIT/latest AUC và ODFV |

Bốn tên ảnh NB1–NB4 là tên bắt buộc của Codelab. Bốn ảnh NB5–NB8 được thêm
để chứng minh trọn khối Advanced.

## 3. Kiểm tra nhanh trước khi chụp

```bash
git status --short
make verify-lite
make test
```

Không chạy lại từng cell rời rạc ngay trước khi chụp, vì có thể làm execution
count và timestamp không còn theo thứ tự. Nếu cần làm mới output, chạy lại
toàn notebook bằng `make notebooks`.
