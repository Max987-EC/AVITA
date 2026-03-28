from flask import Flask, render_template, jsonify
import random
import string

app = Flask(__name__)

# ====== 畫面路由 (負責切換頁面) ======
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/fortune")
def fortune_page():
    return render_template("fortune.html")

@app.route("/password")
def password_page():
    return render_template("password.html")

# ====== API 路由 (負責背後運算) ======
@app.route("/api/get_fortune")
def api_fortune():
    fortunes = ["大吉 🌟", "中吉 ⭐", "小吉 ✨", "凶 🌧️"]
    return jsonify({"result": random.choice(fortunes)})

@app.route("/api/get_password")
def api_password():
    # 產生一組 8 碼的隨機英數密碼
    chars = string.ascii_letters + string.digits
    pwd = ''.join(random.choice(chars) for _ in range(8))
    return jsonify({"result": pwd})

if __name__ == "__main__":
    app.run(debug=True)
