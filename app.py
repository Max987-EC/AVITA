import os
import io
import uuid
import cv2
import numpy as np
import base64
from flask import Flask, request, send_file, render_template, jsonify

# ==========================================
# 📦 自定義模組與第三方套件引入
# ==========================================
from lane_detector import LaneDetector       # 車道線偵測核心邏輯
from moviepy import VideoFileClip            # 用於影片轉碼 (確保網頁能播放)
from image_processor import ImageProcessor   # 綜合影像處理核心邏輯

# 初始化 Flask 應用程式
app = Flask(__name__)

# ==========================================
# 🏠 首頁 (AVITA 大廳)
# ==========================================
@app.route('/')
def index():
    """渲染系統首頁"""
    return render_template('index.html')

# ==========================================
# 🛠️ 工具 1：影像縮放 (Image Resizer)
# ==========================================
@app.route('/tool/image-resizer', methods=['GET', 'POST'])
def image_resizer():
    """處理影像縮放的路由 (目前為保留擴充區塊)"""
    if request.method == 'GET':
        return render_template('image_resizer.html')
        
    if request.method == 'POST':
        pass # 待實作：替換成原本的縮放程式碼

# ==========================================
# 🚗 工具 2：車道線偵測 (Lane Detection)
# ==========================================
@app.route('/tool/lane-detection', methods=['GET', 'POST'])
def lane_detection():
    """接收使用者上傳的影片，進行車道線偵測後回傳處理後的影片"""
    if request.method == 'GET':
        return render_template('lane_detection.html')

    # 1. 檢查是否有上傳影片
    if 'video' not in request.files:
        return "沒有上傳影片", 400
    video_file = request.files['video']
    
    # 2. 建立唯一的暫存檔案名稱 (避免多人同時使用時檔案覆蓋)
    temp_id = uuid.uuid4().hex
    input_path = f"temp_in_{temp_id}.mp4"          # 原始上傳影片
    opencv_output_path = f"temp_cv2_{temp_id}.mp4" # OpenCV 處理後的影片
    web_output_path = f"temp_web_{temp_id}.mp4"    # 轉碼後供網頁播放的影片
    
    video_file.save(input_path)

    # 3. 初始化偵測器與影片讀取物件
    detector = LaneDetector()
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 設定 OpenCV 影片寫入器 (預設輸出兩倍寬度，用於並排對比)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(opencv_output_path, fourcc, fps, (width * 2, height))
    
    # 4. 逐幀處理影片
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break 
        
        # 呼叫車道線偵測邏輯，並寫入新影片
        processed_frame = detector.process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    
    # 5. 影片轉碼 (重要：OpenCV 輸出的 mp4v 編碼網頁無法直接播放，需轉為 H.264)
    try:
        clip = VideoFileClip(opencv_output_path)
        clip.write_videofile(web_output_path, codec="libx264", audio=False, logger=None)
        clip.close()
    except Exception as e:
        return f"影片轉碼失敗: {str(e)}", 500

    # 6. 清理暫存檔案 (釋放伺服器空間)
    if os.path.exists(input_path): os.remove(input_path)
    if os.path.exists(opencv_output_path): os.remove(opencv_output_path)

    # 回傳最終影片給前端
    return send_file(web_output_path, mimetype='video/mp4')

# ==========================================
# 🎨 工具 3：綜合影像處理 (Image Processing)
# ==========================================
@app.route('/tool/image-processing', methods=['GET', 'POST'])
def image_processing():
    """接收圖片與動態參數，執行各種影像濾波與轉換，並回傳 Base64 格式的結果"""
    if request.method == 'GET':
        return render_template('image_processing.html')

    if request.method == 'POST':
        # 1. 檢查圖片是否存在
        if 'image' not in request.files:
            return jsonify({"error": "沒有上傳圖片"}), 400
            
        file = request.files['image']
        process_type = request.form.get('process_type', 'negative') 
        
        # 2. 安全地擷取前端傳來的動態參數 (設定預設值以防萬一)
        threshold = int(request.form.get('threshold', 127))              # 二值化門檻
        c_val = float(request.form.get('c', 1.0))                        # 對數/指數常數
        gamma = float(request.form.get('gamma', 1.0))                    # Gamma 值
        kernel_size = int(request.form.get('kernel_size', 3))            # 卷積核大小
        sigma = float(request.form.get('sigma', 1.0))                    # 高斯標準差
        D0 = float(request.form.get('D0', 30.0))                         # 頻率域截斷頻率
        n_order = int(request.form.get('n', 2))                          # 巴特沃斯階數
        clip_limit = float(request.form.get('clip_limit', 2.0))          # CLAHE 對比限制
        tile_grid_size = int(request.form.get('tile_grid_size', 8))      # CLAHE 網格大小
        threshold1 = int(request.form.get('threshold1', 50))             # Canny 低門檻
        threshold2 = int(request.form.get('threshold2', 150))            # Canny 高門檻
        
        # 🌟 修改：陷波濾波器改為接收獨立的長寬半徑
        notch_d0_u = float(request.form.get('notch_d0_u', 30.0))
        notch_d0_v = float(request.form.get('notch_d0_v', 30.0))
        u0 = int(request.form.get('u0', 50))
        v0 = int(request.form.get('v0', 50))
        
        hough_threshold = int(request.form.get('hough_threshold', 100))
        hough_min_dist = int(request.form.get('hough_min_dist', 20))
        hough_param2 = int(request.form.get('hough_param2', 30))
        hough_min_radius = int(request.form.get('hough_min_radius', 0))
        hough_max_radius = int(request.form.get('hough_max_radius', 0))
        
        # 3. 將上傳的圖片轉換為 OpenCV 可讀取的 Numpy 陣列
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_array is None:
            return jsonify({"error": "圖片讀取失敗"}), 400

        # 初始化影像處理器
        processor = ImageProcessor(img_array)
        
        # ==========================================
        # 4. 根據使用者選擇的模式，執行對應的演算法
        # ==========================================
        
        # [ 基本強度轉換 ]
        if process_type == 'binarize':
            result_img = processor.binarize(threshold)
        elif process_type == 'clahe':  
            result_img = processor.clahe_equalization(clip_limit, tile_grid_size)
        elif process_type == 'negative':
            result_img = processor.negative_transform()
        elif process_type == 'log':
            result_img = processor.log_transform(c_val)
        elif process_type == 'power_law':
            result_img = processor.power_law_transform(gamma, c_val)
        elif process_type == 'equalization':
            result_img = processor.histogram_equalization()
            
        # [ 空間域濾波 ]
        elif process_type == 'mean':
            result_img = processor.mean_filter(kernel_size)
        elif process_type == 'gaussian':
            result_img = processor.gaussian_filter(kernel_size, sigma)
        elif process_type == 'median':
            result_img = processor.median_filter(kernel_size)
            
        # [ 邊緣偵測 ]
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
            
        # [ 頻率域濾波 ]
        elif process_type == 'notch_reject':
            # 🌟 傳入新的長寬參數
            result_img = processor.notch_reject_filter(notch_d0_u, notch_d0_v, u0, v0, n_order)
        elif process_type.startswith('freq_'):
            # 解析字串以決定濾波器類型與高低通 (例如 'freq_butterworth_low')
            parts = process_type.split('_')
            f_type = parts[1] # ideal, butterworth, gaussian
            p_type = parts[2] # low, high
            result_img = processor.frequency_filter(f_type, p_type, D0, n_order)
            
        # 🌟 [ 特徵檢測 (Hough) ]
        elif process_type == 'hough_lines':
            result_img = processor.hough_lines(hough_threshold)
        elif process_type == 'hough_circles':
            # 共用 threshold2 作為 Canny 高門檻 (param1)
            result_img = processor.hough_circles(hough_min_dist, threshold2, hough_param2, hough_min_radius, hough_max_radius)

        # 防呆機制：若無符合條件則回傳原圖
        else:
            result_img = processor.img

        # ==========================================
        # 5. 將所有結果編碼為 Base64 格式準備回傳
        # ==========================================
        
        # 🌟 新增：將原圖也轉成 JPG Base64，解決瀏覽器無法顯示 TIF 的問題
        success_orig, encoded_orig = cv2.imencode('.jpg', img_array)
        orig_b64 = base64.b64encode(encoded_orig).decode('utf-8') if success_orig else ""

        # 主影像編碼 (處理後的圖)
        success, encoded_img = cv2.imencode('.jpg', result_img)
        if not success:
            return jsonify({"error": "影像編碼失敗"}), 500
        processed_b64 = base64.b64encode(encoded_img).decode('utf-8')

        # 產生直方圖 (原圖與處理後)
        orig_hist_b64 = ImageProcessor.generate_histogram_base64(img_array)
        proc_hist_b64 = ImageProcessor.generate_histogram_base64(result_img)

        # 產生傅立葉頻譜圖 (原圖與處理後)
        orig_spec_b64 = ImageProcessor.generate_spectrum_base64(img_array)
        proc_spec_b64 = ImageProcessor.generate_spectrum_base64(result_img)

        # 🌟 新增：將收集到的執行步驟圖片也編碼為 Base64
        steps_b64 = []
        for step_name, step_img in processor.steps:
            _, step_buf = cv2.imencode('.jpg', step_img)
            steps_b64.append({
                "name": step_name,
                "image": base64.b64encode(step_buf).decode('utf-8')
            })

        # 6. 打包成 JSON 格式回傳給前端動態渲染
        return jsonify({
            "original_image": orig_b64,          
            "processed_image": processed_b64,
            "original_histogram": orig_hist_b64,
            "processed_histogram": proc_hist_b64,
            "original_spectrum": orig_spec_b64,  
            "processed_spectrum": proc_spec_b64,
            "steps": steps_b64  # 🌟 新增這行：把步驟資料傳送給前端
        })

# ==========================================
# 🚀 啟動伺服器
# ==========================================
if __name__ == '__main__':
    # debug=True 允許在修改程式碼後自動重啟伺服器
    app.run(debug=True)
