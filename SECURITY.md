# Security & Privacy

- Không commit dữ liệu thô/PII. Thư mục `data/` chỉ dùng local để chạy.
- Identifier phải được **hash** khi xuất public: `id_hash = sha256(id_plain + SALT)`.
- Biến môi trường `SALT` đặt ở file `.env` (không commit).
- Public artifacts chỉ chứa `id_hash`, `score_raw/pd_cal`, decile và số liệu tổng hợp.
