import cv2 
import numpy as np
import time

class CameraAnalyzer:
    def __init__(self, camera_source=0):
        # Kamera başlatma
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            raise ValueError("Kamera açılamadı!")
            
        # Çözünürlük ayarları (isteğe bağlı, performans için düşürülebilir)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Durum değişkenleri
        self.accumulated_frame = None
        self.frame_count = 0
        self.is_calibrating = False
        self.mode = "LIVE" # LIVE, DARK_CALIB, FLAT_CALIB, SPN_VIEW
        
        # Tespit edilen hatalı piksellerin listesi (koordinat olarak)
        self.hot_pixels = []  # Karanlıkta parlayanlar
        self.dead_pixels = [] # Aydınlıkta sönenler
        
        # SPN analizi için matrisler
        self.spn_pattern = None

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def process_calibration(self, frame, mode):
        """
        Kalibrasyon sırasında kareleri toplar ve ortalamasını alır.
        Bu yöntem rastgele gürültüyü (random noise) azaltarak
        sabit gürültüyü (FPN/SPN) ortaya çıkarır.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        if self.accumulated_frame is None:
            self.accumulated_frame = gray
        else:
            # Hareketli ortalama (Running Average)
            cv2.accumulateWeighted(gray, self.accumulated_frame, 0.1)
        
        self.frame_count += 1
        return cv2.convertScaleAbs(self.accumulated_frame)

    def analyze_dark_frame(self, avg_frame, threshold=20):
        """
        Karanlık çerçeve analizi:
        Simsiyah olması gereken yerde parlayan pikselleri (Hot Pixels) bulur.
        """
        # Eşik değerinin üzerindeki pikselleri bul
        _, thresh = cv2.threshold(avg_frame, threshold, 255, cv2.THRESH_BINARY)
        coordinates = cv2.findNonZero(thresh)
        
        detected = []
        if coordinates is not None:
            for coord in coordinates:
                detected.append((coord[0][0], coord[0][1]))
        return detected

    def analyze_flat_frame(self, avg_frame, threshold=200):
        """
        Düz alan (beyaz) analizi:
        Beyaz olması gereken yerde siyah kalan pikselleri (Dead Pixels) bulur.
        """
        # Eşik değerinin altındaki pikselleri bul (Beklenen çok parlak, gelen karanlık)
        # Not: Threshold ortam ışığına göre ayarlanmalıdır.
        _, thresh = cv2.threshold(avg_frame, threshold, 255, cv2.THRESH_BINARY_INV)
        coordinates = cv2.findNonZero(thresh)
        
        detected = []
        if coordinates is not None:
            for coord in coordinates:
                detected.append((coord[0][0], coord[0][1]))
        return detected

    def extract_spn(self, avg_frame):
        """
        Sensor Pattern Noise (SPN) Çıkarımı:
        Basitleştirilmiş yöntem: Görüntüden, görüntünün yumuşatılmış halini çıkarırız.
        Geriye kalan yüksek frekanslı detaylar sensör gürültüsü ve ince dokulardır.
        """
        # 1. Gürültüden arındırılmış (denoised) versiyonu oluştur (Gaussian Blur ile)
        float_avg = avg_frame.astype(np.float32)
        denoised = cv2.GaussianBlur(float_avg, (5, 5), 0)
        
        # 2. Orijinal ortalamadan yumuşatılmış hali çıkar (Geriye gürültü kalır)
        noise_residue = float_avg - denoised
        
        # 3. Görselleştirme için normalize et (0-255 arasına çek ve kontrastı artır)
        # SPN normalde çıplak gözle görülmez, bu yüzden güçlendiriyoruz.
        spn_vis = cv2.normalize(noise_residue, None, 0, 255, cv2.NORM_MINMAX)
        spn_vis = cv2.convertScaleAbs(spn_vis, alpha=5.0, beta=128) # Kontrast artır
        
        return spn_vis

    def run(self):
        print("--- KAMERA SENSÖR ANALİZİ ---")
        print("'d': Karanlık Mod Kalibrasyonu (Lensi Kapatın)")
        print("'f': Düz Alan Kalibrasyonu (Beyaz Bir Yere Tutun)")
        print("'s': SPN (Sensör Gürültüsü) Görünümüne Geç")
        print("'r': Reset / Canlı Mod")
        print("'q': Çıkış")
        
        while True:
            frame = self.capture_frame()
            if frame is None:
                break

            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # Tuş kontrolleri
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d'): # Karanlık Kalibrasyon Başlat
                self.mode = "DARK_CALIB"
                self.accumulated_frame = None
                self.frame_count = 0
                self.hot_pixels = []
                print("Karanlık kalibrasyon başladı. Lütfen lensi kapatın.")
                
            elif key == ord('f'): # Düz Alan Kalibrasyon Başlat
                self.mode = "FLAT_CALIB"
                self.accumulated_frame = None
                self.frame_count = 0
                self.dead_pixels = []
                print("Düz alan kalibrasyonu başladı. Beyaz bir yüzeye tutun.")
            
            elif key == ord('s'): # SPN Görünümü
                if self.accumulated_frame is not None:
                    self.mode = "SPN_VIEW"
                else:
                    print("Önce bir kalibrasyon (d veya f) yaparak veri toplayın.")

            elif key == ord('r'): # Reset
                self.mode = "LIVE"
                self.accumulated_frame = None
                
            elif key == ord('q'): # Çıkış
                break

            # MODLARA GÖRE İŞLEMLER
            if self.mode == "DARK_CALIB":
                avg = self.process_calibration(frame, "dark")
                
                # Anlık analiz (örneğin 30 kare sonra analiz yap)
                if self.frame_count > 30:
                    self.hot_pixels = self.analyze_dark_frame(avg, threshold=15)
                    cv2.putText(display_frame, "Analiz Tamamlandi!", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.mode = "LIVE" # Otomatik olarak canlıya dön
                
                cv2.putText(display_frame, f"Karanlik Kalibrasyon: {self.frame_count}/30", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif self.mode == "FLAT_CALIB":
                avg = self.process_calibration(frame, "flat")
                
                if self.frame_count > 30:
                    # Ortalama parlaklığın %50'sinin altındakileri ölü kabul et
                    mean_val = np.mean(avg)
                    threshold = mean_val * 0.5 
                    self.dead_pixels = self.analyze_flat_frame(avg, threshold=threshold)
                    cv2.putText(display_frame, "Analiz Tamamlandi!", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.mode = "LIVE"
                
                cv2.putText(display_frame, f"Duz Alan Kalibrasyon: {self.frame_count}/30", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            elif self.mode == "SPN_VIEW":
                # Birikmiş kareden SPN çıkar
                if self.accumulated_frame is not None:
                    avg_int = cv2.convertScaleAbs(self.accumulated_frame)
                    spn_vis = self.extract_spn(avg_int)
                    
                    # SPN'i renkliye çevirip ekrana bas (gri tonlamalı)
                    display_frame = cv2.cvtColor(spn_vis, cv2.COLOR_GRAY2BGR)
                    cv2.putText(display_frame, "SPN (Sensor Pattern Noise) Gorunumu", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Canlı modda hatalı pikselleri işaretle
            if self.mode == "LIVE":
                # Hot Pixels (Kırmızı Daire)
                for x, y in self.hot_pixels:
                    cv2.circle(display_frame, (x, y), 5, (0, 0, 255), 1)
                    # Daha belirgin olması için ok işareti de eklenebilir
                
                # Dead Pixels (Mavi Daire)
                for x, y in self.dead_pixels:
                    cv2.circle(display_frame, (x, y), 5, (255, 0, 0), 1)

                info_text = f"Hot: {len(self.hot_pixels)} | Dead: {len(self.dead_pixels)}"
                cv2.putText(display_frame, info_text, (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Sensor Analiz", display_frame)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = CameraAnalyzer()
    app.run()