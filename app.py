import os
from flask import Flask
from flask_session import Session # Flask-Sessionをインポート

# --- Blueprintのインポート ---
from apps.home import home_bp
from apps.lgbm_playground import lgbm_bp

# --- Flaskアプリケーションの生成 ---
app = Flask(__name__)


# --- アプリケーション設定 ---

# セッション管理のためのSECRET_KEY
app.config['SECRET_KEY'] = 'your-secret-key-for-session' # 必ず複雑なキーに変更してください

# --- ▼▼▼ Flask-Sessionの設定 ▼▼▼ ---
# セッションのタイプを 'filesystem' (サーバー上のファイル) に指定
app.config['SESSION_TYPE'] = 'filesystem'
# セッション情報を保存するディレクトリを指定
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_session')
# セッションに署名を行うか
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# アプリにFlask-Sessionを適用
Session(app)
# --- ▲▲▲ ここまで ▲▲▲ ---


# プロジェクトのルートディレクトリを基準にパスを設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# アップロードフォルダとモデル保存フォルダのパスを定義
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

# 定義したフォルダが存在しない場合は、自動で作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True) # セッション用フォルダも作成

# Flaskアプリのコンフィグに設定を保存
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER


# --- Blueprintの登録 ---
app.register_blueprint(home_bp)
app.register_blueprint(lgbm_bp)


# --- 直接実行された場合の設定 ---
if __name__ == '__main__':
    app.run(debug=True)