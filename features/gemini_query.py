import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- GÜVENLİ YAPIlandırma ---
# Proje kök dizinindeki .env dosyasından ortam değişkenlerini yükle
# Bu fonksiyon, GOOGLE_API_KEY gibi hassas bilgileri güvenli bir şekilde okumamızı sağlar.
load_dotenv()

try:
    # .env dosyasından API anahtarını al
    api_key = os.getenv("GOOGLE_API_KEY")

    # Eğer API anahtarı bulunamazsa, kullanıcıyı bilgilendirerek hata ver.
    # Bu, "API key not found" gibi hataları daha en başından çözmemizi sağlar.
    if not api_key:
        raise ValueError("GOOGLE_API_KEY .env dosyasında bulunamadı. Lütfen kontrol edin.")

    # Google Generative AI kütüphanesini yapılandır
    genai.configure(api_key=api_key)

except Exception as e:
    # Yapılandırma sırasında herhangi bir hata olursa, bunu ekrana yazdır.
    print(f"HATA: Google API yapılandırması sırasında bir sorun oluştu: {e}")
    # Uygulamanın çökmemesi için burada bir exit() veya başka bir kontrol mekanizması eklenebilir.

# --- ANA FONKSİYON ---
# Bu fonksiyon, projenin diğer kısımlarından (app.py, agents/ vb.) çağrılacak.
def get_gemini_response(query: str, model_name: str = "gemini-1.5-flash") -> str:
    """
    Verilen bir sorguyu belirtilen Google Gemini modeline gönderir ve metin yanıtını döndürür.

    Args:
        query (str): Kullanıcının sorusu veya modele gönderilecek olan prompt.
        model_name (str): Kullanılacak Gemini modelinin adı. Varsayılan: "gemini-1.5-flash".

    Returns:
        str: Gemini modelinden gelen metin tabanlı yanıt. Hata durumunda bir hata mesajı döndürür.
    """
    try:
        # Belirtilen model adıyla bir üretici model oluştur
        # 'gemini-pro' modeli bazı API versiyonlarında artık desteklenmiyor olabilir.
        # 'gemini-1.5-flash' veya 'gemini-1.5-pro' gibi daha güncel modelleri kullanmak en iyisidir.
        model = genai.GenerativeModel(model_name)
        
        # Sorguyu modele gönder ve yanıtı al
        response = model.generate_content(query)

        # Yanıtın geçerli ve metin içeriğinin dolu olduğundan emin ol
        if response and response.text:
            return response.text
        else:
            # Modelin boş veya geçersiz bir yanıt döndürmesi durumuna karşı önlem
            return "Modelden geçerli bir yanıt alınamadı."

    except Exception as e:
        # API çağrısı sırasında herhangi bir hata olursa (örneğin internet bağlantısı sorunu, geçersiz sorgu vb.)
        error_message = f"API'den yanıt alınırken bir hata oluştu: {e}"
        print(error_message)
        return error_message

# --- DOĞRUDAN ÇALIŞTIRMA VE TEST İÇİN ---
# Bu blok, sadece bu dosyayı terminalden direkt olarak çalıştırdığınızda (örn: python features/gemini_query.py) devreye girer.
# Fonksiyonun doğru çalışıp çalışmadığını hızlıca test etmek için mükemmeldir.
if __name__ == '__main__':
    print("--- gemini_query.py fonksiyon testi ---")

    # Örnek bir sorgu
    test_query = "LangChain nedir ve bir 'agent' ne işe yarar? Kısaca açıkla."
    
    print(f"\nSorgu gönderiliyor: '{test_query}'")

    # Fonksiyonu çağır ve yanıtı al
    gemini_answer = get_gemini_response(test_query)
    
    # Gelen yanıtı ekrana yazdır
    print("\n--- Gemini'den Gelen Yanıt ---")
    print(gemini_answer)
    print("----------------------------")