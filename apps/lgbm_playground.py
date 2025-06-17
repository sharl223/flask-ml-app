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
    current_app, url_for, redirect, jsonify
)

# Blueprintの定義
lgbm_bp = Blueprint(
    'lgbm_playground', __name__,
    template_folder='../templates',
    static_folder='../static'
)

@lgbm_bp.route('/progress/<task_id>')
def progress(task_id):
    """進捗状況を返すAPIエンドポイント"""
    progress_file = os.path.join(current_app.config['UPLOAD_FOLDER'], f'{task_id}.prog')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            message = f.read()
        return jsonify({'message': message})
    return jsonify({'message': '進捗ファイルが見つかりません...'})

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

@lgbm_bp.route('/playground', methods=['GET', 'POST'])
def playground():
    if not current_app.secret_key:
        flash('エラー: アプリケーションにSECRET_KEYが設定されていません。', 'danger')
        return redirect(url_for('home.index'))

    if request.method == 'GET':
        context = session.get('lgbm', {})
        context.setdefault('form_values', {})
        return render_template('lgbm_playground.html', **context)

    action = request.form.get('action')
    context = session.get('lgbm', {})
    context['form_values'] = request.form.to_dict(flat=False)

    try:
        if action == 'upload':
            file = request.files.get('file')
            if not file or file.filename == '':
                flash('ファイルが選択されていません。', 'warning')
                return redirect(request.url)
            
            filename = file.filename
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            context = {'filename': filename}
            context['form_values'] = {}

            df = pd.read_csv(upload_path, index_col=0)
            
            context['columns'] = df.columns.tolist()
            context['df_shape'] = df.shape
            context['df_preview_html'] = df.head().to_html(classes='table table-sm table-striped table-hover', border=0)
            
            default_categoricals = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'bool':
                    default_categoricals.append(col)
                elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 20:
                    series_no_na = df[col].dropna()
                    if not series_no_na.empty and series_no_na.isin([0, 1]).all():
                        default_categoricals.append(col)

            context['default_categoricals'] = default_categoricals
            session['lgbm'] = context
            flash(f'ファイル "{filename}" が正常にアップロードされました。', 'success')
            return redirect(url_for('lgbm_playground.playground'))

        elif action == 'start_optimization':
            # --- ▼▼▼ 修正点: フォームからタスクIDを受け取る ▼▼▼ ---
            task_id = request.form.get('task_id')
            if not task_id:
                flash('タスクIDがありません。', 'danger')
                return redirect(url_for('lgbm_playground.playground'))
            
            progress_file = os.path.join(current_app.config['UPLOAD_FOLDER'], f'{task_id}.prog')

            target_column = request.form['target_column']
            problem_type = request.form['problem_type']
            feature_columns = request.form.getlist('feature_columns')
            categorical_features = request.form.getlist('categorical_features')
            n_trials = int(request.form.get('n_trials', 20))

            df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], context['filename']), index_col=0)
            X, y, _ = _prepare_data(df, target_column, feature_columns, categorical_features)

            def progress_callback(study, trial):
                message = f"最適化を実行中... ({trial.number + 1} / {n_trials} トライアル)"
                with open(progress_file, 'w') as f:
                    f.write(message)

            def objective(trial):
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
                model = lgb.LGBMRegressor(**params) if problem_type == 'regression' else lgb.LGBMClassifier(**params)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=kf, scoring='r2' if problem_type == 'regression' else 'accuracy')
                return np.mean(scores)
            
            with open(progress_file, 'w') as f:
                f.write("最適化を開始します...")

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback], show_progress_bar=False)
            
            context['best_params'] = study.best_params
            context['optuna_results'] = {
                'best_value': study.best_value,
                'best_params_str': str(study.best_params)
            }
            flash('Optunaによる最適化が完了しました。最適なパラメータがセットされました。', 'success')
            
            if os.path.exists(progress_file):
                os.remove(progress_file)

        elif action == 'start_learning':
            params = {
                'learning_rate': float(request.form.get('learning_rate', 0.1)),
                'n_estimators': int(request.form.get('n_estimators', 100)),
                'max_depth': int(request.form.get('max_depth', 7)),
                'num_leaves': int(request.form.get('num_leaves', 31)),
                'subsample': float(request.form.get('subsample', 0.8)),
                'colsample_bytree': float(request.form.get('colsample_bytree', 0.8)),
                'reg_alpha': float(request.form.get('reg_alpha', 0.0)),
                'reg_lambda': float(request.form.get('reg_lambda', 1.0)),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            target_column = request.form['target_column']
            problem_type = request.form['problem_type']
            feature_columns = request.form.getlist('feature_columns')
            categorical_features = request.form.getlist('categorical_features')
            cv_splits = int(request.form.get('cv_splits', 5))

            df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], context['filename']), index_col=0)
            X, y, label_encoders = _prepare_data(df, target_column, feature_columns, categorical_features)
            
            final_feature_names = X.columns.tolist()
            categorical_feature_indices = [final_feature_names.index(col) for col in categorical_features if col in final_feature_names]

            model_params = params.copy()
            model_params['categorical_feature'] = categorical_feature_indices
            
            model = lgb.LGBMRegressor(**model_params) if problem_type == 'regression' else lgb.LGBMClassifier(**model_params)
            
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
            scoring_metric = 'r2' if problem_type == 'regression' else 'accuracy'
            scores = cross_val_score(model, X, y, cv=kf, scoring=scoring_metric)
            
            model.fit(X, y)
            
            model_filename = f"{os.path.splitext(context['filename'])[0]}_model.joblib"
            model_path = os.path.join(current_app.config['MODELS_FOLDER'], model_filename)
            joblib.dump({
                'model': model,
                'feature_columns': feature_columns,
                'categorical_features': categorical_features,
                'target_column': target_column,
                'problem_type': problem_type,
                'label_encoders': label_encoders
            }, model_path)
            
            context['model_path'] = model_path
            context['results'] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'score_metric_name': '決定係数 R2' if problem_type == 'regression' else '正解率 Accuracy',
                'plot_image': _create_feature_importance_plot(model, X.columns)
            }
            flash('モデルの学習が完了しました。', 'success')

        elif action == 'predict':
            predict_file = request.files.get('predict_file')
            if not predict_file:
                flash('予測用のファイルが選択されていません。', 'warning')
                return redirect(request.url)

            saved_data = joblib.load(context['model_path'])
            model = saved_data['model']

            predict_filename = predict_file.filename
            predict_upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], predict_filename)
            predict_file.save(predict_upload_path)
            
            predict_df = pd.read_csv(predict_upload_path, index_col=0)
            
            X_predict, y_true, _ = _prepare_data(predict_df, saved_data['target_column'], saved_data['feature_columns'], saved_data['categorical_features'])
            predictions = model.predict(X_predict)
            
            predict_df[f'predicted_{saved_data["target_column"]}'] = predictions
            context['predict_filename'] = predict_filename
            
            if y_true is not None:
                scorer = r2_score if saved_data['problem_type'] == 'regression' else accuracy_score
                score = scorer(y_true, predictions)
                context['prediction_score'] = {
                    'score': score,
                    'metric_name': '決定係数 R2' if saved_data['problem_type'] == 'regression' else '正解率 Accuracy'
                }
            
            context['prediction_results_html'] = predict_df.to_html(
                classes='table table-striped table-hover', 
                border=0,
                table_id='prediction-table'
            )
            context['prediction_indices'] = predict_df.index.tolist()
            flash('予測が完了しました。', 'info')

        elif action == 'show_shap':
            shap_target_index = int(request.form['shap_target_index'])
            
            saved_data = joblib.load(context['model_path'])
            model = saved_data['model']

            predict_df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], context['predict_filename']), index_col=0)
            X_predict, _, _ = _prepare_data(predict_df, saved_data['target_column'], saved_data['feature_columns'], saved_data['categorical_features'])

            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_predict)

            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[0]
            
            plot_object = shap.force_plot(expected_value, 
                                          shap_values.values[shap_target_index,:], 
                                          X_predict.iloc[shap_target_index,:], 
                                          show=False)
            
            with io.StringIO() as f:
                shap.save_html(f, plot_object, full_html=False)
                context['shap_plot_html'] = f.getvalue()

            if 'shap_js' in context:
                del context['shap_js']

    except Exception as e:
        flash(f'エラーが発生しました: {e}', 'danger')

    session['lgbm'] = context
    return redirect(url_for('lgbm_playground.playground'))