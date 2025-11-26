import cv2
import numpy as np
import gradio as gr

class CameraAnalyzerState:
    def __init__(self):
        self.accumulated_frame = None
        self.frame_count = 0
        self.mode = "LIVE"  # LIVE, DARK_CALIB, FLAT_CALIB, SPN_VIEW
        self.hot_pixels = []
        self.dead_pixels = []
        self.msg = "Hazır. Canlı mod."

    def reset(self):
        self.accumulated_frame = None
        self.frame_count = 0
        self.mode = "LIVE"
        self.msg = "Sıfırlandı."

# Global state (Tek kullanıcı demo için. Çoklu kullanıcı için gr.State kullanılmalı ama basitlik adına global tutuyoruz)
analyzer = CameraAnalyzerState()

def process_frame(frame):
    global analyzer
    
    if frame is None:
        return None

    # Görüntüyü çevir (Mirror effect)
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    # Griye çevir (işlemler için)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- DURUM YÖNETİMİ ---
    if analyzer.mode == "DARK_CALIB" or analyzer.mode == "FLAT_CALIB":
        if analyzer.accumulated_frame is None:
            analyzer.accumulated_frame = gray
        else:
            cv2.accumulateWeighted(gray, analyzer.accumulated_frame, 0.1)
        
        analyzer.frame_count += 1
        
        # 30 kare toplandıysa analizi yap
        if analyzer.frame_count > 30:
            avg_frame = cv2.convertScaleAbs(analyzer.accumulated_frame)
            
            if analyzer.mode == "DARK_CALIB":
                # Hot Pixel Analizi
                _, thresh = cv2.threshold(avg_frame, 25, 255, cv2.THRESH_BINARY)
                coords = cv2.findNonZero(thresh)
                analyzer.hot_pixels = [(p[0][0], p[0][1]) for p in coords] if coords is not None else []
                analyzer.msg = f"Karanlik Kalibrasyon Bitti! {len(analyzer.hot_pixels)} sicak piksel bulundu."
            
            elif analyzer.mode == "FLAT_CALIB":
                # Dead Pixel Analizi
                mean_val = np.mean(avg_frame)
                threshold_val = mean_val * 0.6
                _, thresh = cv2.threshold(avg_frame, threshold_val, 255, cv2.THRESH_BINARY_INV)
                coords = cv2.findNonZero(thresh)
                analyzer.dead_pixels = [(p[0][0], p[0][1]) for p in coords] if coords is not None else []
                analyzer.msg = f"Duz Alan Kalibrasyonu Bitti! {len(analyzer.dead_pixels)} olu piksel bulundu."
            
            # Analiz bitince Live moda dön ama verileri sakla
            analyzer.mode = "LIVE"
            analyzer.accumulated_frame = None
            analyzer.frame_count = 0
        else:
            analyzer.msg = f"Kalibrasyon yapiliyor... {analyzer.frame_count}/30"

    # --- GÖRSELLEŞTİRME ---
    
    # SPN Modu (Gürültü Görme)
    if analyzer.mode == "SPN_VIEW":
        # SPN için anlık kareyi yumuşatıp çıkarıyoruz (basitleştirilmiş anlık SPN)
        # Gerçek SPN birikmiş kare ister ama webcam akışında anlık göstermek daha efektiftir
        float_img = gray
        denoised = cv2.GaussianBlur(float_img, (5, 5), 0)
        residue = float_img - denoised
        spn_vis = cv2.normalize(residue, None, 0, 255, cv2.NORM_MINMAX)
        spn_vis = cv2.convertScaleAbs(spn_vis, alpha=5.0, beta=128)
        display_frame = cv2.cvtColor(spn_vis, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_frame, "SPN MODU", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # İşaretlemeler (Live Modda)
    else:
        # Hot Pixels (Kırmızı)
        for x, y in analyzer.hot_pixels:
            cv2.circle(display_frame, (x, y), 8, (255, 0, 0), 2) # Kırmızı (RGB'de Gradio BGR alabilir dikkat)
            
        # Dead Pixels (Mavi)
        for x, y in analyzer.dead_pixels:
            cv2.circle(display_frame, (x, y), 8, (0, 0, 255), 2) # Mavi

    # Bilgi mesajı
    cv2.putText(display_frame, analyzer.msg, (10, display_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # OpenCV BGR formatından RGB formatına çevir (Gradio için)
    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

def set_mode_dark():
    analyzer.mode = "DARK_CALIB"
    analyzer.accumulated_frame = None
    analyzer.frame_count = 0
    analyzer.msg = "Lensi kapatin..."
    return

def set_mode_flat():
    analyzer.mode = "FLAT_CALIB"
    analyzer.accumulated_frame = None
    analyzer.frame_count = 0
    analyzer.msg = "Beyaz bir duvara tutun..."
    return

def set_mode_spn():
    analyzer.mode = "SPN_VIEW"
    analyzer.msg = "SPN Modu"
    return

def reset_all():
    analyzer.reset()
    return

# --- ARAYÜZ ---
with gr.Blocks() as demo:
    gr.Markdown("# Kamera Dead Pixel & SPN Analizi")
    gr.Markdown("Bu araç tarayıcı kameranızı kullanarak sensör analizi yapar.")
    
    with gr.Row():
        start_dark = gr.Button("1. Karanlık Kalibrasyon (Hot Pixel)")
        start_flat = gr.Button("2. Düz Alan Kalibrasyon (Dead Pixel)")
        start_spn = gr.Button("3. SPN Modu (Gürültü)")
        btn_reset = gr.Button("Sıfırla")

    # Webcam Input (Streaming=True sürekli akış sağlar)
    image_input = gr.Image(sources=["webcam"], streaming=True, label="Kamera")
    image_output = gr.Image(label="Analiz Sonucu")

    # Buton olayları
    start_dark.click(fn=set_mode_dark, inputs=None, outputs=None)
    start_flat.click(fn=set_mode_flat, inputs=None, outputs=None)
    start_spn.click(fn=set_mode_spn, inputs=None, outputs=None)
    btn_reset.click(fn=reset_all, inputs=None, outputs=None)

    # Akış döngüsü
    image_input.stream(fn=process_frame, inputs=image_input, outputs=image_output, time_limit=600)

if __name__ == "__main__":
    demo.launch()