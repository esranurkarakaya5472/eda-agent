<div align="center">

# 🤖 EDA Agent
**CSV yükle → AI analiz etsin, temizlesin, görselleştirsin.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square)](https://streamlit.io)
[![Gemini AI](https://img.shields.io/badge/Gemini_2.5-8E44AD?style=flat-square)](https://ai.google.dev)

</div>

Bir CSV dosyasını otomatik olarak analiz eden, temizleyen ve interaktif dashboard'a dönüştüren otonom bir veri zekası aracı. Google Gemini AI entegrasyonu ile Türkçe yönetici özeti ve veriyle sohbet özelliği sunar.

## Kurulum

```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env
```

## Kullanım

```bash
python -m streamlit run app.py   # Web arayüzü
python main.py veri.csv          # CLI
python main.py --watch           # Klasör izleme modu
```

## Testler

```bash
python -m pytest tests/ -v
```
