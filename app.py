from flask import Flask

# 引入拆分好的 Blueprint 模組
from routes.main import main_bp
from routes.seat import seat_bp
from routes.lane import lane_bp
from routes.image import image_bp

# 初始化 Flask 應用程式
app = Flask(__name__)

# 註冊所有的 Blueprint
app.register_blueprint(main_bp)
app.register_blueprint(seat_bp)
app.register_blueprint(lane_bp)
app.register_blueprint(image_bp)

# ==========================================
# 🚀 啟動伺服器
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)
# ++++