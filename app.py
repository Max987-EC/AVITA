from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

# 首頁：AVITA 大廳
@app.route('/')
def home():
    return render_template('index.html')

# 工具一：影像縮放 (OpenCV 版)
@app.route('/tool/image-resizer', methods=['GET', 'POST'])
def image_resizer():
    if request.method == 'POST':
        # 接收使用者上傳的圖片與設定的長寬
        file = request.files['image']
        width = int(request.form['width'])
        height = int(request.form['height'])

        if file:
            # 1. 將上傳的檔案讀取為位元組 (Bytes)
            in_memory_file = file.read()
            
            # 2. 將位元組轉換為 Numpy 陣列 (OpenCV 的專屬語言)
            np_img = np.frombuffer(in_memory_file, np.uint8)
            
            # 3. 解碼成 OpenCV 影像格式
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            
            # 4. 使用 OpenCV 進行縮放 (INTER_AREA 適合用來縮小圖片，畫質較好)
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            
            # 5. 將處理完的矩陣，重新編碼成 JPG 圖片的位元組
            is_success, buffer = cv2.imencode(".jpg", resized_img)
            io_buf = io.BytesIO(buffer)
            
            # 6. 回傳給使用者下載
            return send_file(
                io_buf, 
                mimetype='image/jpeg', 
                as_attachment=True, 
                download_name='avita_resized.jpg'
            )

    # 如果是 GET 請求，就顯示網頁畫面
    return render_template('image_resizer.html')

if __name__ == '__main__':
    app.run(debug=True)
