# 📷 Kamera Sensör Analizi ve Ölü Piksel Tespiti

Bu proje, Python ve OpenCV kullanarak dijital kamera sensörlerindeki kusurları (ölü pikseller, sıcak pikseller) tespit eden ve sensörün kendine özgü gürültü desenini (Sensor Pattern Noise - SPN) analiz eden bir araçtır.

Proje iki farklı sürüm içerir:

1. **Masaüstü Sürümü**: Yerel bilgisayarınızda yüksek performanslı analiz için.

2. **Web/Gradio Sürümü**: Hugging Face Spaces gibi bulut ortamlarında veya tarayıcı üzerinden çalıştırmak için.

---

# 🚀 Özellikler

**Sıcak Piksel (Hot Pixel) Tespiti**: Karanlık ortamda bile parlak kalan hatalı pikselleri belirler.

**Ölü Piksel (Dead Pixel) Tespiti**: Aydınlık ortamda tepki vermeyen siyah pikselleri belirler.

**SPN (Sensor Pattern Noise) Analizi**: Sensörün üretimden kaynaklanan "parmak izi" sayılabilecek gürültü desenini görselleştirir.

**Gürültü Azaltma**: Rastgele gürültüyü (random noise) elemek için çoklu kare ortalaması (frame averaging) yöntemi kullanır.

---

# 📂 Dosya Yapısı

- `sensor_analysis.py`: (Önerilen) Masaüstü kullanımı içindir. cv2.imshow penceresi açar ve klavye kısayolları ile kontrol edilir.

- `sensor_analysis_gradio.py`: Web arayüzü sürümüdür. Gradio kütüphanesini kullanır ve tarayıcı üzerinden kontrol edilir. Hugging Face Spaces dağıtımı için uygundur.

---

# 🛠️ Kurulum

Öncelikle Python'un yüklü olduğundan emin olun. Ardından projeyi klonlayın ve gerekli kütüphaneleri yükleyin:

``` bash
git clone [https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git)
cd REPO_ADINIZ
pip install -r requirements.txt

```
---

# 💻 Kullanım

1. **Masaüstü Versiyonu (`sensor_analysis.py`)**

Kendi bilgisayarınızda, uygulamayı çalıştırmak için:

``` bash
python sensor_analysis.py
```

**Kontroller:**

- `d:` **Karanlık Kalibrasyon (Dark Frame)** . Lensi kapatın ve bu tuşa basın. Sıcak pikselleri (Hot Pixels) kırmızı ile işaretler.

- `f:` **Düz Alan Kalibrasyonu (Flat Field)** . Kamerayı beyaz bir kağıda/duvara tutun ve bu tuşa basın. Ölü pikselleri (Dead Pixels) mavi ile işaretler.

- `s:` **SPN Modu** . Sensör gürültüsünü görmek için bu tuşa basın.

- `r:` **Reset** . Analizi sıfırlar ve canlı moda döner.

- `q:` **Çıkış** .

2. **Web/Gradio Versiyonu (`sensor_analysis_gradio.py`)**

Tarayıcı üzerinden veya Hugging Face Space üzerinde çalıştırmak için:

``` bash
python sensor_analysis_gradio.py
``` 

Komutu çalıştırdıktan sonra terminalde verilen yerel URL'ye (örneğin http://127.0.0.1:7860) gidin.

---

# 📊 Teknik Detaylar

Bu uygulama, sensör hatalarını tespit etmek için istatistiksel bir yaklaşım kullanır:

1. **Frame Averaging (Kare Ortalaması):** Sensörden gelen anlık görüntüdeki rastgele gürültüyü (shot noise) temizlemek için ardışık 30 karenin ortalaması alınır.

2. **Thresholding (Eşikleme):** 

- Hot Pixel: Ortalama karanlık karede belirli bir eşiğin üzerindeki pikseller işaretlenir.
- Dead Pixel: Ortalama parlak karede ortalamanın çok altında kalan pikseller işaretlenir.

3. **SPN Extraction:** Görüntüden, görüntünün yumuşatılmış (denoised) hali çıkarılarak yüksek frekanslı sensör gürültüsü izole edilir.

---

# 🤝 Katkıda Bulunma

Hataları bildirmek veya özellik eklemek isterseniz lütfen bir "Issue" açın veya "Pull Request" gönderin.

---
# 📄 Lisans

Bu proje MIT Lisansı altında sunulmaktadır.