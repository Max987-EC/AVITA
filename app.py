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
        
        # 1. 從記憶體中直接讀取圖片為 NumPy 陣列
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_array is None:
            return jsonify({"error": "圖片讀取失敗"}), 400

        # 2. 初始化處理器
        processor = ImageProcessor(img_array)
        
        # 3. 根據前端傳來的 process_type 執行對應的處理
        if process_type == 'negative':
            result_img = processor.negative_transform()
        elif process_type == 'equalization':
            result_img = processor.histogram_equalization()
        elif process_type == 'sobel':
            result_img = processor.sobel_filter()
        elif process_type == 'laplacian':
            result_img = processor.laplacian_filter()
        elif process_type == 'freq_gaussian_low':
            result_img = processor.frequency_filter('gaussian', 'low', D0=30)
        elif process_type == 'freq_gaussian_high':
            result_img = processor.frequency_filter('gaussian', 'high', D0=30)
        else:
            result_img = processor.img

        # 4. 將處理後的影像轉為 Base64
        success, encoded_img = cv2.imencode('.jpg', result_img)
        if not success:
            return jsonify({"error": "影像編碼失敗"}), 500
        processed_b64 = base64.b64encode(encoded_img).decode('utf-8')

        # 5. 呼叫 ImageProcessor 的靜態方法產生直方圖 Base64
        # 原圖傳入彩色的 img_array，處理後的圖傳入 result_img
        orig_hist_b64 = ImageProcessor.generate_histogram_base64(img_array)
        proc_hist_b64 = ImageProcessor.generate_histogram_base64(result_img)

        # 6. 將所有資料打包成 JSON 回傳給前端
        return jsonify({
            "processed_image": processed_b64,
            "original_histogram": orig_hist_b64,
            "processed_histogram": proc_hist_b64
        })

# ==========================================
# 啟動伺服器
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)
