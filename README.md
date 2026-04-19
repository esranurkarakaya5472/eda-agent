# EDA Agent
**Otonom Veri Bilimi ve EDA (Keşifçi Veri Analizi) Hattı**

EDA Agent, ham veri setlerini aksiyon alınabilir içgörülere dönüştüren otonom bir araçtır. Veri profilleme, temizleme, görselleştirme ve yapay zeka destekli analiz süreçlerini otomatikleştirir.

###  Temel Yetenekler
* **Otonom Profilleme ve Temizleme:** Aykırı değerleri, eksik verileri otomatik tespit eder ve temizler.
* **Akıllı Gösterge Paneli:** Plotly ile 9+ interaktif, dark-mode grafik.
* **Yapay Zeka Entegrasyonu:** Google Gemini destekli yönetici özetleri ve "verinle sohbet et" özelliği.
* **Üretim Standartlarında:** 67 birim testi (unit test) ile doğrulanmış, sağlam ve ölçeklenebilir altyapı.

###  Teknik Yığın (Tech Stack)
Python | Streamlit | Google Gemini 2.5 | Plotly | Pandas | Pytest

###  Hızlı Başlangıç
1. Bağımlılıkları yükleyin: `pip install -r requirements.txt`
2. `.env` dosyanızı oluşturup `GOOGLE_API_KEY` anahtarınızı ekleyin.
3. Uygulamayı çalıştırın: `streamlit run app.py`
