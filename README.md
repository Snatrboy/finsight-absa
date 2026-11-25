# FinSight ABSA
Python 3.11/3.12 (pilih versi yang mendukung wheel dependensi). Jalankan `pip install -r requirements.txt` lalu `streamlit run app.py`. Versi Torch dan Transformers mengikuti laporan validasi Colab; ketersediaan wheel di host belum diuji.

Buat `.streamlit/secrets.toml` berisi `HF_TOKEN = "token_read_anda"` (jangan commit). Alternatif lokal: set environment `FINSIGHT_MODEL_DIR` ke folder FinSight_models.

GitHub: unggah isi folder ini, termasuk .streamlit/config.toml dan .gitignore, tanpa token. Di share.streamlit.io pilih repo, branch main, app.py, masukkan HF_TOKEN pada Advanced settings > Secrets. Model diambil dari revisi repository private yang tercatat, hanya sesuai pilihan. Model pertama mungkin lambat; RAM gratis belum teruji. Jika BERT gagal karena RAM, gunakan model klasik atau jalankan lokal. Jangan mengubah versi library tanpa pengujian ulang.

Data publik dashboard hanya agregat dan laporan eksperimen, bukan ulasan mentah. Prediksi batch dibatasi 500 baris. Model aktif satu per proses; cache disk Hugging Face dapat bertambah. Toggle tema mengubah halaman dan grafik, widget bawaan dapat mengikuti tema Streamlit. Tidak ada training ulang. Jangan gunakan artefak joblib dari sumber tak tepercaya.
