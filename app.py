import os
import io
import uuid
import cv2
import numpy as np
import base64 # 🌟 新增：用於影像與直方圖的 Base64 編碼
from flask import Flask, request, send_file, render_template, jsonify # 🌟 新增 jsonify

# 👇 引入車道線偵測類別
from lane_detector import LaneDetector

# 👇 引入 MoviePy 來處理影片轉碼
from moviepy import VideoFileClip

# 🌟 新增：引入我們剛剛寫好的影像處理類別
from image_processor import ImageProcessor

# 初始化 Flask 應用程式
app = Flask(__name__)

# ==========================================
# 🏠 首頁 (AVITA 大廳)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# 🛠️ 工具 1：影像縮放 (Image Resizer)
# ==========================================
@app.route('/tool/image-resizer', methods=['GET', 'POST'])
def image_resizer():
    if request.method == 'GET':
        return render_template('image_resizer.html')
        
    if request.method == 'POST':
        pass # 請替換成你原本的程式碼

# ==========================================
# 🚗 工具 2：車道線偵測 (Lane Detection)
# ==========================================
@app.route('/tool/lane-detection', methods=['GET', 'POST'])
def lane_detection():
    if request.method == 'GET':
        return render_template('lane_detection.html')

    if 'video' not in request.files:
        return "沒有上傳影片", 400

    video_file = request.files['video']
    
    temp_id = uuid.uuid4().hex
    input_path = f"temp_in_{temp_id}.mp4"
    opencv_output_path = f"temp_cv2_{temp_id}.mp4"  
    web_output_path = f"temp_web_{temp_id}.mp4"     
    
    video_file.save(input_path)

    detector = LaneDetector()

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(opencv_output_path, fourcc, fps, (width * 2, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
        processed_frame = detector.process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    
    try:
        clip = VideoFileClip(opencv_output_path)
        clip.write_videofile(web_output_path, codec="libx264", audio=False, logger=None)
        clip.close()
    except Exception as e:
        return f"影片轉碼失敗: {str(e)}", 500

    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(opencv_output_path):
        os.remove(opencv_output_path)

    return send_file(web_output_path, mimetype='video/mp4')

# ==========================================
# 🎨 工具 3：綜合影像處理 (Image Processing)
# ==========================================
@app.route('/tool/image-processing', methods=['GET', 'POST'])
def image_processing():
    if request.method == 'GET':
        return render_template('image_processing.html')

    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "沒有上傳圖片"}), 400
            
        file = request.files['image']
        process_type = request.form.get('process_type', 'negative') 
        
        # 🌟 擷取前端傳來的動態參數 (並設定預設值以防萬一)
        threshold = int(request.form.get('threshold', 127))
        c_val = float(request.form.get('c', 1.0))
        gamma = float(request.form.get('gamma', 1.0))
        kernel_size = int(request.form.get('kernel_size', 3))
        sigma = float(request.form.get('sigma', 1.0))
        D0 = float(request.form.get('D0', 30.0))
        n_order = int(request.form.get('n', 2))
        clip_limit = float(request.form.get('clip_limit', 2.0))
        tile_grid_size = int(request.form.get('tile_grid_size', 8))
        
        # 🌟 1. 新增接收 Canny 的兩個參數
        threshold1 = int(request.form.get('threshold1', 50))
        threshold2 = int(request.form.get('threshold2', 150))
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_array is None:
            return jsonify({"error": "圖片讀取失敗"}), 400

        processor = ImageProcessor(img_array)
        
        # 🌟 2. 在 if-elif 判斷式中，加入新的邊緣偵測選項
        if process_type == 'binarize':
            result_img = processor.binarize(threshold)
        elif process_type == 'clahe':  # <--- 新增這兩行
            result_img = processor.clahe_equalization(clip_limit, tile_grid_size)
        elif process_type == 'negative':
            result_img = processor.negative_transform()
        elif process_type == 'log':
            result_img = processor.log_transform(c_val)
        elif process_type == 'power_law':
            result_img = processor.power_law_transform(gamma, c_val)
        elif process_type == 'equalization':
            result_img = processor.histogram_equalization()
        elif process_type == 'mean':
            result_img = processor.mean_filter(kernel_size)
        elif process_type == 'gaussian':
            result_img = processor.gaussian_filter(kernel_size, sigma)
        elif process_type == 'median':
            result_img = processor.median_filter(kernel_size)
            
        # --- 邊緣偵測區塊 ---
        elif process_type == 'roberts':
            result_img = processor.roberts_filter()
        elif process_type == 'prewitt':
            result_img = processor.prewitt_filter()
        elif process_type == 'sobel':
            result_img = processor.sobel_filter()
        elif process_type == 'laplacian':
            result_img = processor.laplacian_filter()
        elif process_type == 'log_edge':
            result_img = processor.log_filter(kernel_size, sigma)
        elif process_type == 'canny':
            result_img = processor.canny_filter(threshold1, threshold2)
        # -------------------
        
        elif process_type.startswith('freq_'):
            # 巧妙解析頻率域的字串 (例如 'freq_butterworth_low')
            parts = process_type.split('_')
            f_type = parts[1] # ideal, butterworth, gaussian
            p_type = parts[2] # low, high
            result_img = processor.frequency_filter(f_type, p_type, D0, n_order)
        else:
            result_img = processor.img

        success, encoded_img = cv2.imencode('.jpg', result_img)
        if not success:
            return jsonify({"error": "影像編碼失敗"}), 500
        processed_b64 = base64.b64encode(encoded_img).decode('utf-8')

        # 產生直方圖
        orig_hist_b64 = ImageProcessor.generate_histogram_base64(img_array)
        proc_hist_b64 = ImageProcessor.generate_histogram_base64(result_img)

        # 🌟 產生傅立葉頻譜圖
        orig_spec_b64 = ImageProcessor.generate_spectrum_base64(img_array)
        proc_spec_b64 = ImageProcessor.generate_spectrum_base64(result_img)

        # 將所有資料打包成 JSON 回傳
        return jsonify({
            "processed_image": processed_b64,
            "original_histogram": orig_hist_b64,
            "processed_histogram": proc_hist_b64,
            "original_spectrum": orig_spec_b64,  # 🌟 新增
            "processed_spectrum": proc_spec_b64  # 🌟 新增
        })

# ==========================================
# 啟動伺服器
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)
