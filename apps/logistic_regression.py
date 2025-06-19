import os
import pandas as pd
import numpy as np
import joblib
import shap
import io
import base64
import matplotlib.pyplot as plt
import japanize_matplotlib
from flask import (
    Blueprint, render_template, request, session, flash,
    current_app, url_for, redirect
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

logistic_regression_bp = Blueprint(
    'logistic_regression', __name__,
    template_folder='../templates',
    static_folder='../static'
)

SESSION_KEY = 'logistic_regression_context'

def _prepare_data_simple(df, feature_columns, numeric_categoricals, scaler=None, train_columns=None):
    """データの前処理。数値の質的変数を指定可能に。"""
    X = df[feature_columns].copy()

    for col in numeric_categoricals:
        if col in X.columns:
            X[col] = X[col].astype('category')

    X = pd.get_dummies(X, drop_first=True, dtype=float)
    if train_columns is not None:
        missing_cols = set(train_columns) - set(X.columns)
        for c in missing_cols:
            X[c] = 0
        X = X[train_columns]
    else:
        train_columns = X.columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns
    if scaler is None:
        scaler = StandardScaler()
        if len(numeric_cols) > 0:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        if len(numeric_cols) > 0:
            X[numeric_cols] = scaler.transform(X[numeric_cols])
    return X, scaler, train_columns

def _create_coefficient_plot(coeffs):
    """モデルの係数を可視化するグラフを生成し、画像データを返す"""
    top_coeffs = pd.concat([coeffs.head(10), coeffs.tail(10)]).sort_values('係数', ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top_coeffs.index, top_coeffs['係数'], color=top_coeffs['係数'].map(lambda x: 'tomato' if x > 0 else 'dodgerblue'))
    plt.title('予測への影響度（係数）', fontsize=18)
    plt.xlabel('係数の大きさ', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

@logistic_regression_bp.route('/logistic_regression', methods=['GET', 'POST'])
def playground():
    if request.method == 'GET':
        context = session.get(SESSION_KEY, {})
        context.setdefault('form_values', {})
        context.setdefault('filename', None)
        context.setdefault('df_shape', None)
        context.setdefault('df_preview_html', None)
        return render_template('logistic_regression.html', **context)

    action = request.form.get('action')
    context = session.get(SESSION_KEY, {})
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
            context = {'filename': filename, 'form_values': {}}
            df = pd.read_csv(upload_path, index_col=0)
            context['columns'] = df.columns.tolist()
            context['df_shape'] = df.shape
            context['df_preview_html'] = df.head().to_html(classes='table table-sm table-striped table-hover', border=0)
            context['numeric_columns'] = df.select_dtypes(include=np.number).columns.tolist()
            flash(f'ファイル "{filename}" が正常にアップロードされました。', 'success')
            
        elif action == 'start_learning':
            target_column = request.form['target_column']
            feature_columns = request.form.getlist('feature_columns')
            numeric_categoricals = request.form.getlist('numeric_categoricals')
            c_value = float(request.form.get('c_value', 1.0))
            
            df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], context['filename']), index_col=0)
            y = df[target_column]
            X, scaler, train_columns = _prepare_data_simple(df, feature_columns, numeric_categoricals)
            
            model = LogisticRegression(random_state=42, C=c_value)
            model.fit(X, y)
            predictions = model.predict(X)
            score = accuracy_score(y, predictions)
            coeffs = pd.DataFrame(model.coef_[0], index=train_columns, columns=['係数']).sort_values('係数', ascending=False)
            context['simple_results'] = {
                'score': score, 'score_metric_name': "正解率 Accuracy",
                'coeffs_plot_image': _create_coefficient_plot(coeffs)
            }
            model_filename = f"logreg_{os.path.splitext(context['filename'])[0]}.joblib"
            model_path = os.path.join(current_app.config['MODELS_FOLDER'], model_filename)
            joblib.dump({'model': model, 'scaler': scaler, 'train_columns': train_columns, 'feature_columns': feature_columns, 'numeric_categoricals': numeric_categoricals, 'target_column': target_column}, model_path)
            context['model_path'] = model_path
            context['X_train_df_for_shap'] = X.to_json(orient='split')
            flash('ロジスティック回帰モデルの学習が完了しました。', 'success')

        elif action == 'predict':
            predict_file = request.files.get('predict_file')
            if not predict_file:
                flash('予測用のファイルが選択されていません。', 'warning')
                return redirect(request.url)
            saved_data = joblib.load(context['model_path'])
            predict_filename = predict_file.filename
            predict_upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], predict_filename)
            predict_file.save(predict_upload_path)
            predict_df = pd.read_csv(predict_upload_path, index_col=0)
            context['predict_filename'] = predict_filename
            y_true = predict_df[saved_data['target_column']] if saved_data['target_column'] in predict_df.columns else None
            
            X_predict, _, _ = _prepare_data_simple(predict_df, saved_data['feature_columns'], saved_data.get('numeric_categoricals', []), scaler=saved_data['scaler'], train_columns=saved_data['train_columns'])
            
            predictions = saved_data['model'].predict(X_predict)
            if y_true is not None:
                score = accuracy_score(y_true, predictions)
                context['prediction_score'] = {'score': score, 'metric_name': "正解率 Accuracy"}
            predict_df[f'predicted_{saved_data["target_column"]}'] = predictions
            context['X_predict_df_for_shap'] = X_predict.to_json(orient='split')
            context['prediction_results_html'] = predict_df.to_html(classes='table table-striped table-hover', border=0, table_id='prediction-table')
            context['prediction_indices'] = predict_df.index.tolist()
            flash('予測が完了しました。', 'info')

        elif action == 'show_shap':
            shap_target_index = int(request.form['shap_target_index'])
            saved_data = joblib.load(context['model_path'])
            model = saved_data['model']
            X_train = pd.read_json(context['X_train_df_for_shap'], orient='split')
            X_predict = pd.read_json(context['X_predict_df_for_shap'], orient='split')
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer(X_predict)
            plot_object = shap.force_plot(explainer.expected_value, shap_values.values[shap_target_index,:], X_predict.iloc[shap_target_index,:], show=False)
            with io.StringIO() as f:
                shap.save_html(f, plot_object, full_html=False)
                context['shap_plot_html'] = f.getvalue()

    except Exception as e:
        flash(f'エラーが発生しました: {e}', 'danger')
        
    session[SESSION_KEY] = context
    return redirect(url_for('logistic_regression.playground'))