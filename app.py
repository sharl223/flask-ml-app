import os
from flask import Flask
from flask_session import Session

# Blueprintのインポート
from apps.home import home_bp
from apps.lgbm_playground import lgbm_bp
from apps.linear_regression import linear_regression_bp
from apps.logistic_regression import logistic_regression_bp

# Flaskアプリケーションの生成
app = Flask(__name__)

# --- アプリケーション設定 ---

# セッション管理のためのSECRET_KEY
app.config['SECRET_KEY'] = 'your-secret-key-for-session'

# Flask-Sessionの設定
app.config['SESSION_TYPE'] = 'filesystem'
# --- ▼▼▼ パス設定の修正 ▼▼▼ ---
# ディスクのマウント先 '/app/data' を基準に各フォルダのパスを定義
DATA_DIR = os.path.join('/app', 'data') if os.environ.get('RENDER') else os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
MODELS_FOLDER = os.path.join(DATA_DIR, 'models')
SESSION_FILE_DIR = os.path.join(DATA_DIR, 'flask_session')
# --- ▲▲▲ パス設定の修正 ▲▲▲ ---

app.config['SESSION_FILE_DIR'] = SESSION_FILE_DIR
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

Session(app)

# 各フォルダが存在しない場合は自動で作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(SESSION_FILE_DIR, exist_ok=True) 

# Flaskアプリのコンフィグに設定を保存
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER

# --- Blueprintの登録 ---
app.register_blueprint(home_bp)
app.register_blueprint(lgbm_bp)
app.register_blueprint(linear_regression_bp)
app.register_blueprint(logistic_regression_bp)

# --- 直接実行された場合の設定 ---
if __name__ == '__main__':
    app.run(debug=True)