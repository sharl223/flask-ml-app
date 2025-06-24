"""
AI Playground - ホームアプリケーション
メインページとサンプルデータの管理
"""

import os
import pandas as pd
from flask import (
    Blueprint, render_template, current_app,
    send_from_directory, make_response, request, flash, redirect, url_for, session
)
from utils import (
    DataProcessor, FileManager, ErrorHandler, ValidationHelper,
    logger, safe_float, safe_int, clear_all_sessions
)

# Blueprintの定義
home_bp = Blueprint(
    'home', __name__,
    template_folder='../templates',
    static_folder='../static'
)

# サンプルデータファイルの設定
SAMPLE_DATASETS = {
    'titanic': {
        'train': 'titanic_train.csv',
        'predict': 'titanic_predict.csv',
        'name': 'タイタニック生存予測',
        'description': '乗客のプロフィールから生存者を予測する分類問題',
        'type': 'classification',
        'records': {'train': 891, 'predict': 418},
        'features': {'train': 12, 'predict': 11}
    },
    'california': {
        'train': 'california_train.csv',
        'predict': 'california_predict.csv',
        'name': 'カリフォルニア住宅価格予測',
        'description': '地域の情報から住宅価格を予測する回帰問題',
        'type': 'regression',
        'records': {'train': 16640, 'predict': 5460},
        'features': {'train': 8, 'predict': 8}
    }
}

@home_bp.route('/')
def index():
    """
    ホームページを表示する
    
    Returns:
        ホームページのテンプレート
    """
    try:
        logger.info("ホームページにアクセスされました")
        return render_template('home.html')
    except Exception as e:
        logger.error(f"ホームページの表示でエラーが発生しました: {e}")
        flash('ページの表示中にエラーが発生しました。', 'danger')
        return render_template('home.html')

@home_bp.route('/reset_all', methods=['POST'])
def reset_all():
    """
    全セッション情報をリセット
    
    Returns:
        ホームページへのリダイレクト
    """
    try:
        # 全セッションをクリア
        cleared_keys = clear_all_sessions(session)
        
        if cleared_keys:
            flash(f'全セッション情報をリセットしました。クリアされた項目: {len(cleared_keys)}個', 'success')
            logger.info(f"全セッションリセット完了: {cleared_keys}")
        else:
            flash('リセットするセッション情報がありませんでした。', 'info')
            logger.info("リセット対象のセッション情報がありませんでした")
        
        return redirect(url_for('home.index'))
        
    except Exception as e:
        logger.error(f"全セッションリセットでエラーが発生しました: {e}")
        flash('リセット処理中にエラーが発生しました。', 'danger')
        return redirect(url_for('home.index'))

@home_bp.route('/datasets')
def datasets():
    """
    サンプルデータページを表示する
    
    Returns:
        サンプルデータページのテンプレート
    """
    try:
        logger.info("サンプルデータページにアクセスされました")
        
        # プロジェクトのルートディレクトリを取得
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        context = {}
        
        # 各サンプルデータセットの情報を取得
        for dataset_key, dataset_info in SAMPLE_DATASETS.items():
            try:
                # 学習用データ
                train_path = os.path.join(root_path, dataset_info['train'])
                if os.path.exists(train_path):
                    train_df = DataProcessor.load_csv_safe(train_path, index_col=0)
                    context[f'{dataset_key}_train_html'] = train_df.head().to_html(
                        classes='table table-sm table-striped', 
                        border=0
                    )
                else:
                    context[f'{dataset_key}_train_html'] = f"ファイルが見つかりません: {dataset_info['train']}"
                
                # 予測用データ
                predict_path = os.path.join(root_path, dataset_info['predict'])
                if os.path.exists(predict_path):
                    predict_df = DataProcessor.load_csv_safe(predict_path, index_col=0)
                    context[f'{dataset_key}_predict_html'] = predict_df.head().to_html(
                        classes='table table-sm table-striped', 
                        border=0
                    )
                else:
                    context[f'{dataset_key}_predict_html'] = f"ファイルが見つかりません: {dataset_info['predict']}"
                    
            except Exception as e:
                logger.error(f"{dataset_key}データセットの読み込みに失敗しました: {e}")
                context[f'{dataset_key}_train_html'] = f"エラー: {e}"
                context[f'{dataset_key}_predict_html'] = f"エラー: {e}"
        
        # データセット情報をコンテキストに追加
        context['datasets'] = SAMPLE_DATASETS
        
        logger.info("サンプルデータの読み込みが完了しました")
        return render_template('datasets.html', **context)
        
    except Exception as e:
        logger.error(f"サンプルデータページの表示でエラーが発生しました: {e}")
        error_message = ErrorHandler.handle_upload_error(e)
        return render_template('datasets.html', error=error_message)

@home_bp.route('/download/<filename>')
def download_file(filename):
    """
    サンプルデータファイルをダウンロードさせる
    
    Args:
        filename: ダウンロードするファイル名
        
    Returns:
        ファイルダウンロードレスポンス
    """
    try:
        # セキュリティチェック：許可されたファイルのみ
        allowed_files = []
        for dataset_info in SAMPLE_DATASETS.values():
            allowed_files.extend([dataset_info['train'], dataset_info['predict']])
        
        # CSV加工集計用サンプルデータも許可
        csv_processor_files = [
            'csv_processor_dataset_01.csv',
            'csv_processor_dataset_02.csv', 
            'csv_processor_dataset_03.csv',
            'csv_processor_dataset_04.csv',
            'csv_processor_dataset_05.csv',
            'csv_processor_dataset_06.csv',
            'csv_processor_dataset_07.csv'
        ]
        allowed_files.extend(csv_processor_files)
        
        if filename not in allowed_files:
            logger.warning(f"許可されていないファイルのダウンロード試行: {filename}")
            return "ファイルが見つかりません。", 404
        
        # プロジェクトのルートディレクトリを取得
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # CSV加工集計用サンプルデータはuploadsフォルダにある
        if filename in csv_processor_files:
            file_path = os.path.join(root_path, 'uploads', filename)
        else:
            file_path = os.path.join(root_path, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"ファイルが見つかりません: {file_path}")
            return "ファイルが見つかりません。", 404
        
        logger.info(f"ファイルダウンロード: {filename}")
        
        # CSV加工集計用サンプルデータはuploadsフォルダから送信
        if filename in csv_processor_files:
            return send_from_directory(os.path.join(root_path, 'uploads'), filename, as_attachment=True)
        else:
            return send_from_directory(root_path, filename, as_attachment=True)
        
    except Exception as e:
        logger.error(f"ファイルダウンロードでエラーが発生しました: {e}")
        return "ダウンロード中にエラーが発生しました。", 500

@home_bp.route('/health')
def health_check():
    """
    ヘルスチェック用エンドポイント
    
    Returns:
        ヘルスステータスのJSONレスポンス
    """
    try:
        # 基本的なシステムチェック
        checks = {
            'app': True,
            'database': True,  # 現在はデータベースを使用していない
            'file_system': True,
            'memory': True
        }
        
        # ファイルシステムのチェック
        try:
            upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder, exist_ok=True)
            
            # 書き込みテスト
            test_file = os.path.join(upload_folder, 'health_check.tmp')
            with open(test_file, 'w') as f:
                f.write('health_check')
            os.remove(test_file)
            
        except Exception as e:
            checks['file_system'] = False
            logger.error(f"ファイルシステムチェックに失敗: {e}")
        
        # メモリチェック
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                checks['memory'] = False
        except ImportError:
            # psutilが利用できない場合はスキップ
            logger.info("psutilが利用できないため、メモリチェックをスキップします")
            pass
        
        # 全体的なステータス
        overall_status = all(checks.values())
        
        response_data = {
            'status': 'healthy' if overall_status else 'unhealthy',
            'timestamp': pd.Timestamp.now().isoformat(),
            'version': '1.0.0',
            'service': 'AI Playground',
            'checks': checks
        }
        
        status_code = 200 if overall_status else 503
        
        logger.info(f"ヘルスチェック実行: {response_data['status']}")
        return response_data, status_code
        
    except Exception as e:
        logger.error(f"ヘルスチェックでエラーが発生しました: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }, 500

@home_bp.route('/api/datasets')
def api_datasets():
    """
    サンプルデータセットのAPIエンドポイント
    
    Returns:
        データセット情報のJSONレスポンス
    """
    try:
        logger.info("API: データセット情報の要求")
        
        # プロジェクトのルートディレクトリを取得
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        api_data = {}
        
        for dataset_key, dataset_info in SAMPLE_DATASETS.items():
            try:
                # 学習用データの情報
                train_path = os.path.join(root_path, dataset_info['train'])
                train_exists = os.path.exists(train_path)
                
                # 予測用データの情報
                predict_path = os.path.join(root_path, dataset_info['predict'])
                predict_exists = os.path.exists(predict_path)
                
                api_data[dataset_key] = {
                    'name': dataset_info['name'],
                    'description': dataset_info['description'],
                    'type': dataset_info['type'],
                    'files': {
                        'train': {
                            'filename': dataset_info['train'],
                            'exists': train_exists,
                            'records': dataset_info['records']['train'],
                            'features': dataset_info['features']['train']
                        },
                        'predict': {
                            'filename': dataset_info['predict'],
                            'exists': predict_exists,
                            'records': dataset_info['records']['predict'],
                            'features': dataset_info['features']['predict']
                        }
                    }
                }
                
            except Exception as e:
                logger.error(f"{dataset_key}データセットのAPI情報取得に失敗: {e}")
                api_data[dataset_key] = {
                    'error': str(e),
                    'name': dataset_info['name']
                }
        
        return api_data
        
    except Exception as e:
        logger.error(f"API: データセット情報の取得でエラーが発生しました: {e}")
        return {'error': str(e)}, 500