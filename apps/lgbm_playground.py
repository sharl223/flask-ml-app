"""
AI Playground - LightGBM Playground
LightGBMを使用した機械学習モデルの学習と予測
"""

import os
import base64
import io
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib
import uuid
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from flask import (
    Blueprint, render_template, request, session, flash,
    current_app, url_for, redirect, jsonify, send_from_directory
)
from utils import (
    DataProcessor, ModelManager, VisualizationHelper, 
    FileManager, ErrorHandler, ValidationHelper,
    logger, safe_float, safe_int, generate_unique_filename
)

# Blueprintの定義
lgbm_bp = Blueprint(
    'lgbm_playground', __name__,
    template_folder='../templates',
    static_folder='../static'
)

# セッションキー
SESSION_KEY = 'lgbm'

@lgbm_bp.route('/progress/<task_id>')
def progress(task_id):
    """
    進捗状況を返すAPIエンドポイント
    
    Args:
        task_id: タスクID
        
    Returns:
        進捗状況のJSON
    """
    try:
        progress_file = os.path.join(current_app.config['UPLOAD_FOLDER'], f'{task_id}.prog')
        logger.info(f"進捗確認リクエスト - タスクID: {task_id}, ファイル: {progress_file}")
        
        if os.path.exists(progress_file):
            # ファイルの最終更新時刻を確認
            file_mtime = os.path.getmtime(progress_file)
            logger.info(f"進捗ファイル最終更新時刻: {file_mtime}")
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                message = f.read()
            logger.info(f"進捗ファイル読み取り成功: {message[:100]}...")
            
            # キャッシュを無効化するヘッダーを設定
            response = jsonify({'message': message})
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        else:
            logger.warning(f"進捗ファイルが見つかりません: {progress_file}")
            return jsonify({'message': '進捗ファイルが見つかりません...'})
            
    except Exception as e:
        logger.error(f"進捗取得でエラーが発生しました: {e}")
        return jsonify({'message': '進捗の取得に失敗しました...'})

def _prepare_data(df, target_column, feature_columns, categorical_features):
    """データの前処理（ラベルエンコーディング）を行う内部関数"""
    df_processed = df[feature_columns].copy()
    
    label_encoders = {}
    for col in categorical_features:
        if col in df_processed.columns:
            le = LabelEncoder()
            valid_data = df_processed[col][df_processed[col].notna()]
            le.fit(valid_data)
            df_processed[col] = df_processed[col].map(lambda s: le.transform([s])[0] if pd.notna(s) and s in le.classes_ else s)
            label_encoders[col] = le
            
    X = df_processed
    y = df[target_column] if target_column in df.columns else None

    return X, y, label_encoders

def _create_feature_importance_plot(model, feature_names):
    """特徴量の重要度グラフを生成し、画像データを返す内部関数"""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('予測に重要な影響を与えた情報 Top 20', fontsize=18)
    plt.xlabel('重要度', fontsize=14)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

@lgbm_bp.route('/lgbm_playground', methods=['GET', 'POST'])
def playground():
    """
    LightGBMプレイグラウンドのメインルート
    
    Returns:
        プレイグラウンドページのテンプレート
    """
    # SECRET_KEYの確認
    if not current_app.secret_key:
        flash('エラー: アプリケーションにSECRET_KEYが設定されていません。', 'danger')
        return redirect(url_for('home.index'))

    if request.method == 'GET':
        context = session.get(SESSION_KEY, {})
        # リセット後の確実な初期化
        if not context:
            context = {
                'form_values': {},
                'filename': None,
                'df_shape': None,
                'df_preview_html': None,
                'optuna_results': None,
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
            context.setdefault('optuna_results', None)
            context.setdefault('simple_results', None)
            context.setdefault('prediction_results_html', None)
            context.setdefault('shap_plot_html', None)
            context.setdefault('prediction_score', None)
            context.setdefault('prediction_indices', [])
        
        return render_template('lgbm_playground.html', **context)

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
            
        elif action == 'start_optimization':
            context = handle_optimization(context)
            
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
                'optuna_results': None,
                'simple_results': None,
                'prediction_results_html': None,
                'shap_plot_html': None,
                'prediction_score': None,
                'prediction_indices': []
            }
            flash('LightGBMデータがリセットされました。', 'info')
            return redirect(url_for('lgbm_playground.playground'))
            
        else:
            flash('無効なアクションです。', 'warning')

    except Exception as e:
        error_message = ErrorHandler.handle_model_error(e)
        flash(f'エラーが発生しました: {error_message}', 'danger')
        logger.error(f"LightGBMプレイグラウンドでエラーが発生しました: {e}")

    session[SESSION_KEY] = context
    return redirect(url_for('lgbm_playground.playground'))

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
        
        # カテゴリカル特徴量の自動検出
        context['default_categoricals'] = DataProcessor.detect_categorical_features(df)
        
        # 数値列の検出（数値の質的変数選択用）
        numeric_columns = []
        recommended_numeric_categoricals = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # カテゴリカル特徴量として既に検出されている列は除外
                if col in context['default_categoricals']:
                    logger.info(f"数値の質的変数から除外（既にカテゴリカル特徴量）: {col}")
                    continue
                
                # すべての数値列を選択可能にする
                numeric_columns.append(col)
                
                # 推奨候補の検出（一意値が少ない数値列）
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                
                # 条件1: 一意値が50以下（より少ない値に変更）
                # 条件2: 一意値の割合が30%以下（より緩い条件に変更）
                # 条件3: 一意値が2以上（完全に連続的でない）
                if (unique_count <= 50 and 
                    unique_count / total_count <= 0.3 and 
                    unique_count >= 2):
                    recommended_numeric_categoricals.append(col)
                    logger.info(f"数値の質的変数推奨候補として検出: {col} (一意値: {unique_count}, 割合: {unique_count/total_count:.3f})")
        
        context['numeric_columns'] = numeric_columns
        context['recommended_numeric_categoricals'] = recommended_numeric_categoricals
        logger.info(f"数値の質的変数選択可能列: {numeric_columns}")
        logger.info(f"数値の質的変数推奨候補: {recommended_numeric_categoricals}")
        
        # デフォルトのパラメータ値を設定
        context['form_values'] = {
            'learning_rate': ['0.1'],
            'n_estimators': ['100'],
            'max_depth': ['7'],
            'num_leaves': ['31'],
            'subsample': ['0.8'],
            'colsample_bytree': ['0.8'],
            'reg_alpha': ['0.0'],
            'reg_lambda': ['1.0'],
            'problem_type': ['regression']
        }
        
        flash(f'ファイル "{filename}" が正常にアップロードされました。', 'success')
        logger.info(f"ファイルアップロード完了: {filename}")
        
        return context
        
    except Exception as e:
        error_message = ErrorHandler.handle_upload_error(e)
        flash(error_message, 'warning')
        raise

def handle_optimization(context):
    """
    Optunaによるハイパーパラメータ最適化を処理
    
    Args:
        context: 現在のコンテキスト
        
    Returns:
        更新されたコンテキスト
    """
    try:
        logger.info("=== 最適化処理開始 ===")
        
        # タスクIDの生成
        task_id = request.form.get('task_id') or str(uuid.uuid4())
        progress_file = os.path.join(current_app.config['UPLOAD_FOLDER'], f'{task_id}.prog')
        
        logger.info(f"タスクID: {task_id}")
        logger.info(f"進捗ファイル: {progress_file}")
        
        # 進捗ファイルの初期化
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write("パラメータ最適化を開始します...")
        
        # フォームデータの取得
        target_column = request.form['target_column']
        problem_type = request.form['problem_type']
        feature_columns = request.form.getlist('feature_columns')
        categorical_features = request.form.getlist('categorical_features')
        numeric_categoricals = request.form.getlist('numeric_categoricals')
        n_trials = safe_int(request.form.get('n_trials', 20), 20)
        
        logger.info(f"最適化開始 - タスクID: {task_id}, 試行回数: {n_trials}")
        logger.info(f"目的変数: {target_column}")
        logger.info(f"問題タイプ: {problem_type}")
        logger.info(f"特徴量数: {len(feature_columns)}")
        logger.info(f"カテゴリカル特徴量: {categorical_features}")
        logger.info(f"数値の質的変数: {numeric_categoricals}")
        
        # 数値の質的変数をカテゴリカル特徴量に追加
        all_categorical_features = categorical_features + numeric_categoricals
        
        # データの読み込みと前処理
        logger.info("データ読み込み開始...")
        df = DataProcessor.load_csv_safe(
            os.path.join(current_app.config['UPLOAD_FOLDER'], context['filename']), 
            index_col=0
        )
        logger.info(f"データ読み込み完了 - 形状: {df.shape}")
        
        logger.info("データ前処理開始...")
        X, y, _ = DataProcessor.prepare_data_for_lgbm(
            df, target_column, feature_columns, all_categorical_features
        )
        logger.info(f"データ前処理完了 - X形状: {X.shape}, y形状: {y.shape}")
        logger.info(f"特徴量名: {list(X.columns)}")
        
        # 進捗コールバック
        def progress_callback(study, trial):
            try:
                current_trial = trial.number + 1
                best_value = study.best_value if study.best_value is not None else 0.0
                message = f"パラメータ最適化を実行中... ({current_trial} / {n_trials} トライアル)\n現在のベストスコア: {best_value:.4f}"
                
                logger.info(f"進捗コールバック実行 - トライアル {current_trial}, ベストスコア: {best_value:.4f}")
                
                with open(progress_file, 'w', encoding='utf-8') as f:
                    f.write(message)
                
                logger.info(f"最適化進捗 - トライアル {current_trial}/{n_trials}, ベストスコア: {best_value:.4f}")
                logger.info(f"進捗ファイル更新: {progress_file}")
                
            except Exception as e:
                logger.error(f"進捗コールバックでエラー: {e}")
                logger.error(f"進捗コールバックエラーの詳細: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"進捗コールバックスタックトレース: {traceback.format_exc()}")
                # エラーが発生しても最適化は続行
                pass
        
        # 目的関数
        def objective(trial):
            try:
                current_trial = trial.number + 1
                logger.info(f"トライアル {current_trial} 開始")
                
                # トライアル開始時の進捗更新
                best_value = study.best_value if study.best_value is not None else 0.0
                message = f"パラメータ最適化を実行中... ({current_trial} / {n_trials} トライアル)\n現在のベストスコア: {best_value:.4f}"
                with open(progress_file, 'w', encoding='utf-8') as f:
                    f.write(message)
                logger.info(f"目的関数内進捗更新 - トライアル {current_trial}, ベストスコア: {best_value:.4f}")
                
                params = {
                    'objective': 'regression' if problem_type == 'regression' else 'binary',
                    'metric': 'rmse' if problem_type == 'regression' else 'logloss',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
                
                logger.info(f"トライアル {current_trial} パラメータ: {params}")
                
                model = lgb.LGBMRegressor(**params) if problem_type == 'regression' else lgb.LGBMClassifier(**params)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scoring = 'r2' if problem_type == 'regression' else 'accuracy'
                scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
                
                score = scores.mean()
                logger.info(f"トライアル {current_trial} 完了 - スコア: {score:.4f}")
                
                # トライアル完了時の進捗更新
                best_value = study.best_value if study.best_value is not None else 0.0
                message = f"パラメータ最適化を実行中... ({current_trial} / {n_trials} トライアル)\n現在のベストスコア: {best_value:.4f}"
                with open(progress_file, 'w', encoding='utf-8') as f:
                    f.write(message)
                logger.info(f"目的関数内進捗更新完了 - トライアル {current_trial}, ベストスコア: {best_value:.4f}")
                
                return score
                
            except Exception as e:
                logger.error(f"目的関数でエラーが発生: {e}")
                logger.error(f"目的関数エラーの詳細: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"目的関数スタックトレース: {traceback.format_exc()}")
                # エラーが発生した場合は低いスコアを返す
                return -999.0
        
        # 最適化の開始
        logger.info("Optuna最適化開始...")
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write("パラメータ最適化を開始します...")
        
        study = optuna.create_study(direction='maximize')
        logger.info("Study作成完了")
        
        logger.info(f"最適化実行開始 - 試行回数: {n_trials}")
        study.optimize(
            objective, 
            n_trials=n_trials, 
            callbacks=[progress_callback], 
            show_progress_bar=False
        )
        logger.info("最適化実行完了")
        
        # 結果の保存
        logger.info(f"最適化結果 - ベストスコア: {study.best_value}")
        logger.info(f"最適化結果 - ベストパラメータ: {study.best_params}")
        
        context['best_params'] = study.best_params
        context['optuna_results'] = {
            'best_value': study.best_value,
            'best_params_str': str(study.best_params)
        }
        
        # 進捗ファイルの削除
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        flash('Optunaによる最適化が完了しました。最適なパラメータがセットされました。', 'success')
        logger.info("Optuna最適化完了")
        logger.info("=== 最適化処理終了 ===")
        
        return context
        
    except Exception as e:
        logger.error(f"最適化処理でエラーが発生しました: {e}")
        logger.error(f"エラーの詳細: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"スタックトレース: {traceback.format_exc()}")
        
        # エラーが発生した場合でも進捗ファイルを削除
        try:
            if os.path.exists(progress_file):
                os.remove(progress_file)
                logger.info(f"エラー後の進捗ファイル削除完了: {progress_file}")
        except Exception as cleanup_error:
            logger.warning(f"進捗ファイルの削除に失敗: {cleanup_error}")
        
        logger.info("=== 最適化処理エラー終了 ===")
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
        problem_type = request.form['problem_type']
        feature_columns = request.form.getlist('feature_columns')
        categorical_features = request.form.getlist('categorical_features')
        numeric_categoricals = request.form.getlist('numeric_categoricals')
        cv_splits = safe_int(request.form.get('cv_splits', 5), 5)
        
        # 数値の質的変数をカテゴリカル特徴量に追加
        all_categorical_features = categorical_features + numeric_categoricals
        
        # データの読み込みと前処理
        df = DataProcessor.load_csv_safe(
            os.path.join(current_app.config['UPLOAD_FOLDER'], context['filename']), 
            index_col=0
        )
        X, y, label_encoders = DataProcessor.prepare_data_for_lgbm(
            df, target_column, feature_columns, all_categorical_features
        )
        
        # カテゴリカル特徴量のインデックス設定
        final_feature_names = X.columns.tolist()
        categorical_feature_indices = [
            final_feature_names.index(col) 
            for col in all_categorical_features 
            if col in final_feature_names
        ]
        
        # モデルの作成と学習
        model_params = {
            'learning_rate': safe_float(request.form.get('learning_rate', 0.1)),
            'n_estimators': safe_int(request.form.get('n_estimators', 100)),
            'max_depth': safe_int(request.form.get('max_depth', 7)),
            'num_leaves': safe_int(request.form.get('num_leaves', 31)),
            'subsample': safe_float(request.form.get('subsample', 0.8)),
            'colsample_bytree': safe_float(request.form.get('colsample_bytree', 0.8)),
            'reg_alpha': safe_float(request.form.get('reg_alpha', 0.0)),
            'reg_lambda': safe_float(request.form.get('reg_lambda', 1.0)),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        model_params['categorical_feature'] = categorical_feature_indices
        
        model = lgb.LGBMRegressor(**model_params) if problem_type == 'regression' else lgb.LGBMClassifier(**model_params)
        
        # クロスバリデーション
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scoring_metric = 'r2' if problem_type == 'regression' else 'accuracy'
        scores = cross_val_score(model, X, y, cv=kf, scoring=scoring_metric)
        
        # モデルの学習
        model.fit(X, y)
        
        # モデルの保存
        model_filename = f"{os.path.splitext(context['filename'])[0]}_model.joblib"
        model_data = {
            'model': model,
            'feature_columns': feature_columns,
            'categorical_features': categorical_features,
            'numeric_categoricals': numeric_categoricals,
            'target_column': target_column,
            'problem_type': problem_type,
            'label_encoders': label_encoders
        }
        
        model_path = ModelManager.save_model(
            model_data, 
            model_filename, 
            current_app.config['MODELS_FOLDER']
        )
        
        # 特徴量重要度プロットの生成
        plot_image = VisualizationHelper.create_feature_importance_plot(model, X.columns)
        
        # 結果の保存
        context['model_path'] = model_path
        context['results'] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'score_metric_name': '決定係数 R2' if problem_type == 'regression' else '正解率 Accuracy',
            'plot_image': plot_image
        }
        
        flash('モデルの学習が完了しました。', 'success')
        logger.info("モデル学習完了")
        
        return context
        
    except Exception as e:
        logger.error(f"モデル学習でエラーが発生しました: {e}")
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
        model = saved_data['model']
        
        # 予測データの読み込みと前処理
        predict_df = DataProcessor.load_csv_safe(predict_upload_path, index_col=0)
        
        # 数値の質的変数をカテゴリカル特徴量に追加
        all_categorical_features = saved_data.get('categorical_features', []) + saved_data.get('numeric_categoricals', [])
        
        X_predict, y_true, _ = DataProcessor.prepare_data_for_lgbm(
            predict_df, 
            saved_data['target_column'], 
            saved_data['feature_columns'], 
            all_categorical_features
        )
        
        # 予測の実行
        predictions = model.predict(X_predict)
        predict_df[f'predicted_{saved_data["target_column"]}'] = predictions
        
        # 予測結果をファイルに保存
        output_filename = f"predicted_{predict_filename}"
        output_path = os.path.join(current_app.config['UPLOAD_FOLDER'], output_filename)
        predict_df.to_csv(output_path, index=True)
        
        # 結果の保存
        context['predict_filename'] = predict_filename
        context['predicted_filename'] = output_filename
        
        # 評価指標の計算（ターゲットが存在する場合）
        if y_true is not None:
            scorer = r2_score if saved_data['problem_type'] == 'regression' else accuracy_score
            score = scorer(y_true, predictions)
            context['prediction_score'] = {
                'score': score,
                'metric_name': '決定係数 R2' if saved_data['problem_type'] == 'regression' else '正解率 Accuracy'
            }
        
        context['prediction_results_html'] = predict_df.head(5).to_html(
            classes='table table-striped table-hover', 
            border=0,
            table_id='prediction-table'
        )
        context['prediction_indices'] = predict_df.index.tolist()
        
        flash('予測が完了しました。', 'info')
        logger.info("予測完了")
        
        return context
        
    except Exception as e:
        logger.error(f"予測処理でエラーが発生しました: {e}")
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
        
        # 予測データの読み込みと前処理
        predict_df = DataProcessor.load_csv_safe(
            os.path.join(current_app.config['UPLOAD_FOLDER'], context['predict_filename']), 
            index_col=0
        )
        
        # 数値の質的変数をカテゴリカル特徴量に追加
        all_categorical_features = saved_data.get('categorical_features', []) + saved_data.get('numeric_categoricals', [])
        
        X_predict, _, _ = DataProcessor.prepare_data_for_lgbm(
            predict_df, 
            saved_data['target_column'], 
            saved_data['feature_columns'], 
            all_categorical_features
        )
        
        # SHAP分析の実行
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_predict)
        
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[0]
        
        # SHAPプロットの生成
        plot_object = shap.force_plot(
            expected_value, 
            shap_values.values[shap_target_index,:], 
            X_predict.iloc[shap_target_index,:], 
            show=False
        )
        
        with io.StringIO() as f:
            shap.save_html(f, plot_object, full_html=False)
            context['shap_plot_html'] = f.getvalue()
        
        # SHAPのJavaScriptを削除（重複を避けるため）
        if 'shap_js' in context:
            del context['shap_js']
        
        logger.info("SHAP分析完了")
        return context
        
    except Exception as e:
        logger.error(f"SHAP分析でエラーが発生しました: {e}")
        raise

def handle_download_prediction(context):
    """
    Prediction results download
    
    Args:
        context: 現在のコンテキスト
        
    Returns:
        Downloaded file
    """
    try:
        # 予測結果のファイルを取得
        predicted_file = context.get('predicted_filename')
        if not predicted_file:
            flash('予測結果のファイルが見つかりません。', 'warning')
            return redirect(url_for('lgbm_playground.playground'))
        
        # ファイルのパスを取得
        predicted_path = os.path.join(current_app.config['UPLOAD_FOLDER'], predicted_file)
        if not os.path.exists(predicted_path):
            flash('予測結果のファイルが見つかりません。', 'warning')
            return redirect(url_for('lgbm_playground.playground'))
        
        # 予測結果ファイルをダウンロード
        return send_from_directory(current_app.config['UPLOAD_FOLDER'], predicted_file, as_attachment=True)
        
    except Exception as e:
        logger.error(f"予測結果のダウンロードでエラーが発生しました: {e}")
        raise