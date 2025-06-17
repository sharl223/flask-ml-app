import os
import pandas as pd
from flask import (
    Blueprint, render_template, current_app,
    send_from_directory, make_response
)

home_bp = Blueprint(
    'home', __name__,
    template_folder='../templates',
    static_folder='../static'
)

@home_bp.route('/')
def index():
    """ホームページを表示する"""
    return render_template('home.html')

@home_bp.route('/datasets')
def datasets():
    """サンプルデータページを表示する"""
    try:
        # プロジェクトのルートディレクトリからファイルパスを生成
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        titanic_train_path = os.path.join(root_path, 'titanic_train.csv')
        titanic_predict_path = os.path.join(root_path, 'titanic_predict.csv')
        california_train_path = os.path.join(root_path, 'california_train.csv')
        california_predict_path = os.path.join(root_path, 'california_predict.csv')

        # 各CSVファイルを読み込み、先頭5行だけをHTMLに変換
        titanic_train_df = pd.read_csv(titanic_train_path, index_col=0).head()
        titanic_predict_df = pd.read_csv(titanic_predict_path, index_col=0).head()
        california_train_df = pd.read_csv(california_train_path, index_col=0).head()
        california_predict_df = pd.read_csv(california_predict_path, index_col=0).head()

        context = {
            'titanic_train_html': titanic_train_df.to_html(classes='table table-sm table-striped', border=0),
            'titanic_predict_html': titanic_predict_df.to_html(classes='table table-sm table-striped', border=0),
            'california_train_html': california_train_df.to_html(classes='table table-sm table-striped', border=0),
            'california_predict_html': california_predict_df.to_html(classes='table table-sm table-striped', border=0),
        }
    except Exception as e:
        context = {'error': f"サンプルデータの読み込みに失敗しました: {e}"}

    return render_template('datasets.html', **context)

@home_bp.route('/download/<filename>')
def download_file(filename):
    """サンプルデータファイルをダウンロードさせる"""
    try:
        # プロジェクトのルートディレクトリを取得して、そこからファイルを送信
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return send_from_directory(root_path, filename, as_attachment=True)
    except FileNotFoundError:
        return "ファイルが見つかりません。", 404

@home_bp.route('/sitemap.xml')
def sitemap():
    """sitemap.xmlを返すためのルート"""
    try:
        # プロジェクトのルートディレクトリを取得
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # send_from_directory を使って安全にファイルを返す
        response = make_response(send_from_directory(root_path, 'sitemap.xml'))
        # 正しいContent-Typeヘッダーを設定
        response.headers['Content-Type'] = 'application/xml'
        return response
    except FileNotFoundError:
        return "サイトマップが見つかりません。", 404