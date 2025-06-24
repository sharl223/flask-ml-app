"""
AI Playground - ロジスティック回帰
ロジスティック回帰モデルの学習と予測
"""

import os
import pandas as pd
import numpy as np
import shap
import io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import (
    Blueprint, render_template, request, session, flash,
    current_app, url_for, redirect, send_from_directory
)
from utils import (
    DataProcessor, ModelManager, VisualizationHelper, 
    FileManager, ErrorHandler, ValidationHelper,
    logger, safe_float, safe_int
)

# Blueprintの定義
logistic_regression_bp = Blueprint(
    'logistic_regression', __name__,
    template_folder='../templates',
    static_folder='../static'
)

# セッションキー
SESSION_KEY = 'logistic_regression_context'

@logistic_regression_bp.route('/logistic_regression', methods=['GET', 'POST'])
def playground():
    """
    ロジスティック回帰プレイグラウンドのメインルート
    
    Returns:
        プレイグラウンドページのテンプレート
    """
    if request.method == 'GET':
        context = session.get(SESSION_KEY, {})
        # リセット後の確実な初期化
        if not context:
            context = {
                'form_values': {},
                'filename': None,
                'df_shape': None,
                'df_preview_html': None,
                'columns': [],
                'numeric_columns': [],
                'simple_results': None,
                'prediction_results_html': None,
                'shap_plot_html': None,
                'prediction_score': None,
                'prediction_indices': []
            }
        else:
            # 既存のコンテキストにデフォルト値を設定
            context.setdefault('form_values', {})
            context.setdefault('filename', None)
            context.setdefault('df_shape', None)
            context.setdefault('df_preview_html', None)
            context.setdefault('columns', [])
            context.setdefault('numeric_columns', [])
            context.setdefault('simple_results', None)
            context.setdefault('prediction_results_html', None)
            context.setdefault('shap_plot_html', None)
            context.setdefault('prediction_score', None)
            context.setdefault('prediction_indices', [])
        
        return render_template('logistic_regression.html', **context)

    action = request.form.get('action')
    context = session.get(SESSION_KEY, {})
    
    # uploadアクションの場合は既存のform_valuesを保持、それ以外は新しいform_valuesを設定
    if action == 'upload':
        # uploadアクションの場合は既存のform_valuesを保持
        context['form_values'] = context.get('form_values', {})
    else:
        # 他のアクションの場合は新しいform_valuesを設定
        context['form_values'] = request.form.to_dict(flat=False)

    try:
        if action == 'upload':
            context = handle_file_upload(context)
            
        elif action == 'start_learning':
            context = handle_model_training(context)
            
        elif action == 'predict':
            context = handle_prediction(context)
            
        elif action == 'show_shap':
            context = handle_shap_analysis(context)
            
        elif action == 'download_prediction':
            return handle_download_prediction(context)
            
        elif action == 'reset':
            # データリセット
            session.pop(SESSION_KEY, None)
            # セッションファイルも削除を試行
            try:
                if 'SESSION_FILE_DIR' in current_app.config:
                    session_dir = current_app.config['SESSION_FILE_DIR']
                    if os.path.exists(session_dir):
                        import glob
                        session_files = glob.glob(os.path.join(session_dir, '*'))
                        for file in session_files:
                            try:
                                os.remove(file)
                                logger.info(f"セッションファイルを削除しました: {file}")
                            except Exception as e:
                                logger.warning(f"セッションファイルの削除に失敗: {file}, エラー: {e}")
            except Exception as e:
                logger.warning(f"セッションファイル削除でエラー: {e}")
            
            # 空のコンテキストを設定して確実にリセット
            session[SESSION_KEY] = {
                'form_values': {},
                'filename': None,
                'df_shape': None,
                'df_preview_html': None,
                'simple_results': None,
                'prediction_results_html': None,
                'shap_plot_html': None
            }
            flash('ロジスティック回帰データがリセットされました。', 'info')
            return redirect(url_for('logistic_regression.playground'))
            
        else:
            flash('無効なアクションです。', 'warning')

    except Exception as e:
        error_message = ErrorHandler.handle_model_error(e)
        flash(f'エラーが発生しました: {error_message}', 'danger')
        logger.error(f"ロジスティック回帰プレイグラウンドでエラーが発生しました: {e}")

    session[SESSION_KEY] = context
    return redirect(url_for('logistic_regression.playground'))

def handle_file_upload(context):
    """
    ファイルアップロードを処理
    
    Args:
        context: 現在のコンテキスト
        
    Returns:
        更新されたコンテキスト
    """
    try:
        file = request.files.get('file')
        upload_path = FileManager.save_uploaded_file(
            file, 
            current_app.config['UPLOAD_FOLDER']
        )
        
        filename = os.path.basename(upload_path)
        context = {'filename': filename, 'form_values': {}}
        
        # CSVファイルの読み込みと検証
        df = DataProcessor.load_csv_safe(upload_path, index_col=0)
        
        # データの基本情報を設定
        context['columns'] = df.columns.tolist()
        context['df_shape'] = df.shape
        context['df_preview_html'] = df.head().to_html(
            classes='table table-sm table-striped table-hover', 
            border=0
        )
        context['numeric_columns'] = df.select_dtypes(include=np.number).columns.tolist()
        
        flash(f'ファイル "{filename}" が正常にアップロードされました。', 'success')
        logger.info(f"ファイルアップロード完了: {filename}")
        
        return context
        
    except Exception as e:
        error_message = ErrorHandler.handle_upload_error(e)
        flash(error_message, 'warning')
        raise

def handle_model_training(context):
    """
    モデルの学習を処理
    
    Args:
        context: 現在のコンテキスト
        
    Returns:
        更新されたコンテキスト
    """
    try:
        # フォームデータの取得
        target_column = request.form['target_column']
        feature_columns = request.form.getlist('feature_columns')
        numeric_categoricals = request.form.getlist('numeric_categoricals')
        c_value = safe_float(request.form.get('c_value', 1.0), 1.0)
        
        # データの読み込みと前処理
        df = DataProcessor.load_csv_safe(
            os.path.join(current_app.config['UPLOAD_FOLDER'], context['filename']), 
            index_col=0
        )
        y = df[target_column]
        X, scaler, train_columns = DataProcessor.prepare_data_for_linear(
            df, feature_columns, numeric_categoricals
        )
        
        # モデルの作成と学習
        model = LogisticRegression(random_state=42, C=c_value)
        model.fit(X, y)
        
        # 予測と評価
        predictions = model.predict(X)
        score = accuracy_score(y, predictions)
        
        # 係数の分析
        coeffs = pd.DataFrame(
            model.coef_[0], 
            index=train_columns, 
            columns=['係数']
        ).sort_values('係数', ascending=False)
        
        # 係数プロットの生成
        coeffs_plot_image = VisualizationHelper.create_coefficient_plot(coeffs)
        
        # 結果の保存
        context['simple_results'] = {
            'score': score, 
            'score_metric_name': "正解率 Accuracy",
            'coeffs_plot_image': coeffs_plot_image
        }
        
        # モデルの保存
        model_filename = f"logreg_{os.path.splitext(context['filename'])[0]}.joblib"
        model_data = {
            'model': model, 
            'scaler': scaler, 
            'train_columns': train_columns, 
            'feature_columns': feature_columns, 
            'numeric_categoricals': numeric_categoricals, 
            'target_column': target_column
        }
        
        model_path = ModelManager.save_model(
            model_data, 
            model_filename, 
            current_app.config['MODELS_FOLDER']
        )
        
        context['model_path'] = model_path
        context['X_train_df_for_shap'] = X.to_json(orient='split')
        
        flash('ロジスティック回帰モデルの学習が完了しました。', 'success')
        logger.info("ロジスティック回帰モデル学習完了")
        
        return context
        
    except Exception as e:
        logger.error(f"ロジスティック回帰モデル学習でエラーが発生しました: {e}")
        raise

def handle_prediction(context):
    """
    予測を処理
    
    Args:
        context: 現在のコンテキスト
        
    Returns:
        更新されたコンテキスト
    """
    try:
        predict_file = request.files.get('predict_file')
        if not predict_file:
            flash('予測用のファイルが選択されていません。', 'warning')
            return context
        
        # 予測ファイルの保存
        predict_upload_path = FileManager.save_uploaded_file(
            predict_file, 
            current_app.config['UPLOAD_FOLDER']
        )
        predict_filename = os.path.basename(predict_upload_path)
        
        # モデルの読み込み
        saved_data = ModelManager.load_model(context['model_path'])
        
        # 予測データの読み込みと前処理
        predict_df = DataProcessor.load_csv_safe(predict_upload_path, index_col=0)
        context['predict_filename'] = predict_filename
        
        # ターゲット列の存在確認
        y_true = predict_df[saved_data['target_column']] if saved_data['target_column'] in predict_df.columns else None
        
        # 予測データの前処理
        X_predict, _, _ = DataProcessor.prepare_data_for_linear(
            predict_df, 
            saved_data['feature_columns'], 
            saved_data.get('numeric_categoricals', []), 
            scaler=saved_data['scaler'], 
            train_columns=saved_data['train_columns']
        )
        
        # 予測の実行
        predictions = saved_data['model'].predict(X_predict)
        
        # 評価指標の計算（ターゲットが存在する場合）
        if y_true is not None:
            score = accuracy_score(y_true, predictions)
            context['prediction_score'] = {
                'score': score, 
                'metric_name': "正解率 Accuracy"
            }
        
        # 予測結果の追加
        predict_df[f'predicted_{saved_data["target_column"]}'] = predictions
        context['X_predict_df_for_shap'] = X_predict.to_json(orient='split')
        context['prediction_results_html'] = predict_df.head(5).to_html(
            classes='table table-striped table-hover', 
            border=0, 
            table_id='prediction-table'
        )
        context['prediction_indices'] = predict_df.index.tolist()
        
        flash('予測が完了しました。', 'info')
        logger.info("ロジスティック回帰予測完了")
        
        return context
        
    except Exception as e:
        logger.error(f"ロジスティック回帰予測でエラーが発生しました: {e}")
        raise

def handle_shap_analysis(context):
    """
    SHAP分析を処理
    
    Args:
        context: 現在のコンテキスト
        
    Returns:
        更新されたコンテキスト
    """
    try:
        shap_target_index = safe_int(request.form['shap_target_index'], 0)
        
        # モデルの読み込み
        saved_data = ModelManager.load_model(context['model_path'])
        model = saved_data['model']
        
        # 学習データと予測データの読み込み
        X_train = pd.read_json(context['X_train_df_for_shap'], orient='split')
        X_predict = pd.read_json(context['X_predict_df_for_shap'], orient='split')
        
        # SHAP分析の実行
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer(X_predict)
        
        # SHAPプロットの生成
        plot_object = shap.force_plot(
            explainer.expected_value, 
            shap_values.values[shap_target_index,:], 
            X_predict.iloc[shap_target_index,:], 
            show=False
        )
        
        with io.StringIO() as f:
            shap.save_html(f, plot_object, full_html=False)
            context['shap_plot_html'] = f.getvalue()
        
        logger.info("ロジスティック回帰SHAP分析完了")
        return context
        
    except Exception as e:
        logger.error(f"ロジスティック回帰SHAP分析でエラーが発生しました: {e}")
        raise

def handle_download_prediction(context):
    """
    予測結果のダウンロードを処理
    
    Args:
        context: 現在のコンテキスト
        
    Returns:
        ダウンロードファイル
    """
    try:
        # 予測結果のファイルを取得
        predict_file = context.get('predict_filename')
        if not predict_file:
            flash('予測結果のファイルが見つかりません。', 'warning')
            return redirect(url_for('logistic_regression.playground'))
        
        # ファイルのパスを取得
        predict_path = os.path.join(current_app.config['UPLOAD_FOLDER'], predict_file)
        if not os.path.exists(predict_path):
            flash('予測結果のファイルが見つかりません。', 'warning')
            return redirect(url_for('logistic_regression.playground'))
        
        # モデルの読み込み
        saved_data = ModelManager.load_model(context['model_path'])
        
        # 予測データの読み込みと前処理
        predict_df = DataProcessor.load_csv_safe(predict_path, index_col=0)
        
        # 予測データの前処理
        X_predict, _, _ = DataProcessor.prepare_data_for_linear(
            predict_df, 
            saved_data['feature_columns'],
            saved_data.get('numeric_categoricals', []),
            scaler=saved_data['scaler'], 
            train_columns=saved_data['train_columns']
        )
        
        # 予測の実行
        predictions = saved_data['model'].predict(X_predict)
        
        # 予測結果の追加
        predict_df[f'predicted_{saved_data["target_column"]}'] = predictions
        
        # 予測結果ファイルの保存
        output_filename = f"predicted_{predict_file}"
        output_path = os.path.join(current_app.config['UPLOAD_FOLDER'], output_filename)
        predict_df.to_csv(output_path, index=True)
        
        # ファイルをダウンロード
        return send_from_directory(current_app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)
        
    except Exception as e:
        logger.error(f"予測結果のダウンロードでエラーが発生しました: {e}")
        raise