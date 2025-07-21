from dotenv import load_dotenv
import os
import google.generativeai as genai

# .env dosyasını yükle
load_dotenv()

# .env içinden API key al
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("HATA: API key yüklenemedi!")
    exit(1)

genai.configure(api_key=api_key)

# Kullanıcıdan metin alıp Gemini'den cevap döndüren fonksiyon
def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-2.5-pro')  # Doğru model adı
    response = model.generate_content(prompt)
    return response.text

# Test amaçlı çalıştırma
if __name__ == "__main__":
    try:
        cevap = get_gemini_response(
            "Dünyadaki lastik markalarından satışı en çok olan 5 lastik markasının ismi ve satış adedi nedir?"
        )
        print(cevap)
    except Exception as e:
        print(f"Hata: {e}")
        print("Model adını 'gemini-2.5-pro' ile deneyebilirsiniz.")