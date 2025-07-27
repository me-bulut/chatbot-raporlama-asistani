# 🤖 Chatbot Destekli Raporlama Asistanı

Doğal dil ile konuşarak veritabanlarından ve çeşitli dosyalardan raporlar oluşturan, analiz eden ve görselleştiren bir yapay zekâ asistanı.


---

## 📝 Projenin Amacı

Bu proje, teknik bilgisi olmayan kullanıcıların bile karmaşık veri setleriyle kolayca etkileşime girmesini sağlamayı hedefler. Geleneksel raporlama araçlarında saatler sürebilecek veri analizi ve görselleştirme işlemleri, bu asistan sayesinde saniyeler içinde, basit bir sohbet arayüzü üzerinden gerçekleştirilebilir. Kullanıcıların SQL sorguları yazma veya karmaşık Excel formülleriyle uğraşma ihtiyacını ortadan kaldırır.

## ✨ Temel Özellikler

-   **Doğal Dil Anlama:** Kullanıcılardan gelen "Geçen ayki satışları ürün kategorisine göre grupla" gibi karmaşık istekleri anlar.
-   **Otomatik Veri İşleme:** Excel, CSV gibi dosyalardan veya doğrudan PostgreSQL veritabanından veri çekebilir.
-   **Akıllı Analiz ve Özetleme:** Çekilen verileri analiz eder, istatistiksel özetler çıkarır ve metinsel olarak yorumlar.
-   **Dinamik Grafik Oluşturma:** Analiz sonuçlarını `matplotlib` ve `plotly` kullanarak interaktif grafiklere dönüştürür.
-   **Agent Mimarisi:** `LangChain` kullanarak görevleri planlayabilen ve doğru araçları (veri okuma, analiz etme, grafik çizme) otonom olarak seçebilen bir "agent" yapısı kullanır.
-   **RAG Entegrasyonu (Geliştirme Aşamasında):** PDF gibi dökümanlardan bağlamsal bilgi çekerek raporları daha da zenginleştirir.

## 🛠️ Kullanılan Teknolojiler

-   **Backend:** Python
-   **Web Arayüzü:** Streamlit
-   **Yapay Zekâ & Dil Modelleri:** Google Gemini, LangChain
-   **Veri İşleme:** Pandas, NumPy
-   **Veri Görselleştirme:** Matplotlib, Plotly
-   **Veritabanı:** PostgreSQL (psycopg2-binary ile)

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### 1. Projeyi Klonlayın

```bash
git clone https://github.com/me-bulut/chatbot-raporlama-asistani.git
cd chatbot-raporlama-asistani
```

### 2. Sanal Ortam Oluşturun ve Aktif Edin

Proje bağımlılıklarını sisteminizden izole tutmak için bir sanal ortam oluşturun.

```bash
# Sanal ortamı oluştur
python -m venv .venv

# Sanal ortamı aktif et (Windows - Git Bash)
source .venv/Scripts/activate

# Sanal ortamı aktif et (macOS/Linux)
# source .venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

Projenin ihtiyaç duyduğu tüm kütüphaneleri `requirements.txt` dosyasından yükleyin.

```bash
pip install -r requirements.txt
```

### 4. Ortam Değişkenlerini Ayarlayın

Projenin ana dizininde `.env` adında bir dosya oluşturun. Bu dosya, API anahtarlarınızı içerecektir.

```
GOOGLE_API_KEY="BURAYA_KENDİ_GEMINI_API_ANAHTARINIZI_GİRİN"
```
> **Not:** `.env` dosyası, güvenlik nedeniyle `.gitignore` dosyasına eklenmiştir ve GitHub'a gönderilmez.

### 5. Uygulamayı Başlatın

Her şey hazır! Aşağıdaki komutla Streamlit uygulamasını başlatın.

```bash
streamlit run app.py
```
Uygulama, varsayılan web tarayıcınızda otomatik olarak açılacaktır.

## 📖 Nasıl Kullanılır?

1.  Uygulama arayüzü açıldığında, metin giriş kutusunu göreceksiniz.
2.  Analiz etmek istediğiniz veriyle ilgili sorunuzu doğal dilde yazın.
    -   *Örnek: "Excel dosyasındaki verilerin özetini çıkar."*
    -   *Örnek: "Bölgelere göre toplam kar marjını gösteren bir bar grafiği çiz."*
3.  Enter'a basın ve asistanın raporunuzu hazırlamasını bekleyin.

---
