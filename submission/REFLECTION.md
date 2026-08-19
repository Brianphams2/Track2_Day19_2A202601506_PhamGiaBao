# Reflection — Lab 19

**Tên:** Phạm Gia Bảo  
**MSSV:** 2A202601506  
**Cohort:** A20-K4  
**Path:** Lite

Theo output đang lưu trong `02_hybrid_search_rrf.ipynb`, Hybrid đạt
Precision@10 tổng thể 80,8%, cao hơn Vector 80,4% và BM25 77,8%. Với
`exact`, Hybrid đạt 98,0%; khi dùng một bộ truy hồi, BM25 đạt 96,7% và
cao hơn Vector 95,3%, vì các thuật ngữ kỹ thuật xuất hiện nguyên văn. Với
`paraphrase`, Vector đạt 43,3%, cao hơn Hybrid 41,3% và BM25 33,3%, do biểu
diễn vector giữ được ý nghĩa khi thay đổi cách diễn đạt. Với `mixed`, Hybrid
đạt 97,5%, cao hơn hai mode thuần (cùng 97,0%), vì RRF kết hợp cả bằng chứng
lexical lẫn semantic.

Em chọn BM25 khi truy vấn cần khớp chính xác mã, tên kỹ thuật hoặc cần độ
trễ thấp và dễ giải thích. Và chọn Vector khi truy vấn có paraphrase, đa
ngôn ngữ hoặc ít token trùng lặp. Hybrid phù hợp làm mặc định cho truy vấn
hỗn hợp, code-switching và sai chính tả vì giảm rủi ro bỏ sót từ cả hai cách
truy hồi.

## Điều ngạc nhiên nhất khi làm lab này

Điều làm em bất ngờ là Vector thắng rõ ở nhóm `paraphrase`, còn Hybrid chỉ
nhỉnh hơn 0,4 điểm phần trăm trên toàn bộ golden set; Hybrid không mặc nhiên
thắng ở mọi lát cắt truy vấn.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: Không
