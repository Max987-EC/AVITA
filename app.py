import os
import uuid
import cv2
from flask import Flask, request, send_file, render_template

# 👇 引入我們剛剛寫好的車道線偵測類別
from lane_detector import LaneDetector

# 初始化 Flask 應用程式
app = Flask(__name__)

# ==========================================
# 🏠 首頁 (AVITA 大廳)
# ==========================================
@app.route('/')
def index():
    # 這裡回傳你的大廳頁面 (假設叫做 index.html)
    return render_template('index.html')

# ==========================================
# 🛠️ 工具 1：影像縮放 (Image Resizer)
# ==========================================
@app.route('/tool/image-resizer', methods=['GET', 'POST'])
def image_resizer():
    if request.method == 'GET':
        return render_template('image_resizer.html')
        
    if request.method == 'POST':
        # 這裡放你原本處理影像縮放的 Python 邏輯
        # (接收圖片、讀取寬高、cv2.resize、儲存暫存檔、send_file...)
        pass # 請替換成你原本的程式碼

# ==========================================
# 🚗 工具 2：車道線偵測 (Lane Detection)
# ==========================================
@app.route('/tool/lane-detection', methods=['GET', 'POST'])
def lane_detection():
    # 如果是 GET 請求，回傳網頁畫面
    if request.method == 'GET':
        return render_template('lane_detection.html')

    # 如果是 POST 請求，開始處理上傳的影片
    if 'video' not in request.files:
        return "沒有上傳影片", 400

    video_file = request.files['video']
    
    # 產生隨機檔名，避免檔案覆蓋
    temp_id = uuid.uuid4().hex
    input_path = f"temp_in_{temp_id}.mp4"
    output_path = f"temp_out_{temp_id}.mp4"
    
    video_file.save(input_path)

    # 🌟 建立專屬的偵測器實體，避免多人同時使用時記憶體打架
    detector = LaneDetector()

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 使用網頁支援度較高的 avc1 (H.264) 編碼
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
        # 使用專屬偵測器來處理每一幀
        processed_frame = detector.process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    
    # 處理完畢後，刪除原始輸入檔以節省空間
    if os.path.exists(input_path):
        os.remove(input_path)

    # 將處理好的影片傳回給前端
    return send_file(output_path, as_attachment=True, download_name="lane_detected.mp4")

# ==========================================
# 啟動伺服器
# ==========================================
if __name__ == '__main__':
    # debug=True 可以在你修改程式碼時自動重新啟動伺服器
    app.run(debug=True)
