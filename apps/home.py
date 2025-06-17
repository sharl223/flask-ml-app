from flask import Blueprint, render_template, send_from_directory
import pandas as pd
import os

# Blueprintオブジェクトを作成
home_bp = Blueprint('home', __name__, template_folder='../../templates', static_folder='../../static')

# このファイルの場所を基準に、一つ上の階層(プロジェクトルート)のパスを取得
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

@home_bp.route('/')
def home():
    return render_template('home.html')

@home_bp.route('/datasets')
def datasets():
    try:
        titanic_train_df = pd.read_csv(os.path.join(DATA_DIR, 'titanic_train.csv'), index_col=0).head()
        titanic_predict_df = pd.read_csv(os.path.join(DATA_DIR, 'titanic_predict.csv'), index_col=0).head()
        california_train_df = pd.read_csv(os.path.join(DATA_DIR, 'california_train.csv'), index_col=0).head()
        california_predict_df = pd.read_csv(os.path.join(DATA_DIR, 'california_predict.csv'), index_col=0).head()

        # pandas DataFrameをHTMLテーブルに変換
        # classes属性でCSSからデザインを適用できるようにする
        context = {
            'titanic_train_html': titanic_train_df.to_html(classes='dataframe', border=0),
            'titanic_predict_html': titanic_predict_df.to_html(classes='dataframe', border=0),
            'california_train_html': california_train_df.to_html(classes='dataframe', border=0),
            'california_predict_html': california_predict_df.to_html(classes='dataframe', border=0)
        }
    except FileNotFoundError:
        context = { 'error': 'CSVファイルが見つかりません。データセット生成スクリプトを実行してください。' }

    return render_template('datasets.html', **context)


@home_bp.route('/download/<filename>')
def download_file(filename):
    safe_files = [
        "titanic_train.csv", "titanic_predict.csv",
        "california_train.csv", "california_predict.csv"
    ]
    if filename in safe_files:
        return send_from_directory(DATA_DIR, filename, as_attachment=True)
    else:
        return "File not found.", 404