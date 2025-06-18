import os
import pandas as pd
from flask import (
    Blueprint, render_template, request, session, flash,
    current_app, url_for, redirect
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Blueprintの定義
logistic_regression_bp = Blueprint(
    'logistic_regression', __name__,
    template_folder='../templates',
    static_folder='../static'
)

def _prepare_data_simple(df, feature_columns):
    """データの前処理（ダミー変数化とスケーリング）"""
    X = df[feature_columns].copy()
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X, scaler

@logistic_regression_bp.route('/logistic_regression', methods=['GET', 'POST'])
def playground():
    if request.method == 'GET':
        lgbm_context = session.get('lgbm', {})
        return render_template('logistic_regression.html', **lgbm_context)

    action = request.form.get('action')
    context = session.get('lgbm', {})
    context['form_values'] = request.form.to_dict(flat=False)

    try:
        if action == 'start_learning':
            target_column = request.form['target_column']
            feature_columns = request.form.getlist('feature_columns')

            df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], context['filename']), index_col=0)
            y = df[target_column]
            X, scaler = _prepare_data_simple(df, feature_columns)
            final_feature_names = X.columns.tolist()

            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            
            predictions = model.predict(X)
            score = accuracy_score(y, predictions)

            coeffs = pd.DataFrame(
                model.coef_[0],
                index=final_feature_names,
                columns=['係数']
            ).sort_values('係数', ascending=False)
            
            context['simple_results'] = {
                'score': score,
                'score_metric_name': "正解率 Accuracy",
                'coeffs_html': coeffs.to_html(classes='table table-sm table-striped table-hover', border=0)
            }
            flash('ロジスティック回帰モデルの学習が完了しました。', 'success')

    except Exception as e:
        flash(f'エラーが発生しました: {e}', 'danger')
        
    session['lgbm'] = context
    return redirect(url_for('logistic_regression.playground'))