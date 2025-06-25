"""
AI Playground - メインアプリケーション
機械学習モデルの学習と予測を行うWebアプリケーション
"""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify, make_response
from flask_session import Session
from utils import logger, cleanup_temp_files, setup_logging

# Blueprintのインポート
from apps.home import home_bp
from apps.lgbm_playground import lgbm_bp
from apps.linear_regression import linear_regression_bp
from apps.logistic_regression import logistic_regression_bp
from apps.csv_processor import csv_processor_bp

def create_app(config_name: str = None) -> Flask:
    """
    Flaskアプリケーションのファクトリ関数
    
    Args:
        config_name: 設定名（開発/本番環境など）
        
    Returns:
        設定済みのFlaskアプリケーション
    """
    app = Flask(__name__)
    
    # 基本設定
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-for-session')
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1時間
    app.config['SESSION_FILE_THRESHOLD'] = 500  # セッションファイルの閾値
    
    # 環境に応じたパス設定
    if os.environ.get('RENDER'):
        # Render環境
        DATA_DIR = '/app/data'
    else:
        # ローカル環境
        DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ディレクトリ設定
    UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
    MODELS_FOLDER = os.path.join(DATA_DIR, 'models')
    SESSION_FILE_DIR = os.path.join(DATA_DIR, 'flask_session')
    
    # ディレクトリの作成
    for folder in [UPLOAD_FOLDER, MODELS_FOLDER, SESSION_FILE_DIR]:
        os.makedirs(folder, exist_ok=True)
    
    # Flask設定に保存
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MODELS_FOLDER'] = MODELS_FOLDER
    app.config['SESSION_FILE_DIR'] = SESSION_FILE_DIR
    
    # セッション初期化
    Session(app)
    
    # Blueprintの登録
    app.register_blueprint(home_bp)
    app.register_blueprint(lgbm_bp)
    app.register_blueprint(linear_regression_bp)
    app.register_blueprint(logistic_regression_bp)
    app.register_blueprint(csv_processor_bp)
    
    # エラーハンドラー
    register_error_handlers(app)
    
    # アプリケーション開始時の処理
    with app.app_context():
        logger.info("AI Playground アプリケーションを開始します")
        
        # 一時ファイルのクリーンアップ
        cleanup_count = cleanup_temp_files(UPLOAD_FOLDER)
        if cleanup_count > 0:
            logger.info(f"{cleanup_count}個の一時ファイルをクリーンアップしました")
    
    # SEO対策用のヘッダー設定
    @app.after_request
    def add_seo_headers(response):
        """SEO対策用のHTTPヘッダーを追加"""
        # セキュリティヘッダー
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # パフォーマンスヘッダー
        response.headers['Cache-Control'] = 'public, max-age=3600'
        
        # SEOヘッダー
        response.headers['X-Robots-Tag'] = 'index, follow'
        
        return response

    # サイトマップ
    @app.route('/sitemap.xml')
    def sitemap():
        """XMLサイトマップを提供"""
        response = make_response(send_file('sitemap.xml'))
        response.headers['Content-Type'] = 'application/xml'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response

    # robots.txt
    @app.route('/robots.txt')
    def robots():
        """robots.txtを提供"""
        response = make_response("""User-agent: *
Allow: /

# Sitemap
Sitemap: https://flask-ml-app-g0mo.onrender.com/sitemap.xml""")
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response

    # プライバシーポリシー
    @app.route('/privacy')
    def privacy():
        """プライバシーポリシーページ"""
        return render_template('privacy.html')

    # 利用規約
    @app.route('/terms')
    def terms():
        """利用規約ページ"""
        return render_template('terms.html')

    # ヘルスチェック
    @app.route('/health')
    def health_check():
        """ヘルスチェック用エンドポイント"""
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'service': 'AI Playground'
        })
    
    return app

def register_error_handlers(app: Flask) -> None:
    """
    エラーハンドラーを登録
    
    Args:
        app: Flaskアプリケーション
    """
    @app.errorhandler(404)
    def not_found_error(error):
        """404エラーハンドラー"""
        logger.warning(f"404エラー: {request.url}")
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """500エラーハンドラー"""
        logger.error(f"500エラー: {error}")
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(413)
    def too_large(error):
        """ファイルサイズ超過エラーハンドラー"""
        logger.warning("ファイルサイズが上限を超えました")
        return render_template('errors/413.html'), 413

# アプリケーションインスタンスの作成
app = create_app()

# 直接実行された場合の設定
if __name__ == '__main__':
    # 開発環境での設定
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    logger.info(f"開発サーバーを開始します: {host}:{port}, デバッグモード: {debug_mode}")
    
    app.run(
        host=host,
        port=port,
        debug=debug_mode,
        threaded=True
    )