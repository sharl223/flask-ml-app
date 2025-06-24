"""
AI Playground - CSV加工集計
CSVデータの前処理、加工、集計機能を提供

このモジュールは以下の機能を提供します：
- CSVファイルのアップロードと検証
- データの前処理（欠損値処理、データ型変換）
- インデックス処理
- 集計処理とピボットテーブル
- 処理結果のダウンロード
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from flask import (
    Blueprint, render_template, request, session, flash,
    current_app, url_for, redirect, jsonify, send_file
)
from utils import (
    DataProcessor, FileManager, ErrorHandler, ValidationHelper,
    logger, safe_float, safe_int, generate_unique_filename, clear_all_sessions
)
import io

# Blueprintの定義
csv_processor_bp = Blueprint(
    'csv_processor', __name__,
    template_folder='../templates',
    static_folder='../static'
)

# セッションキー
SESSION_KEY = 'csv_processor_context'

# ===== ヘルパー関数 =====

def get_cell_class(value, dtype):
    """
    セルのCSSクラスを取得
    
    Args:
        value: セルの値
        dtype: データ型
        
    Returns:
        CSSクラス名
    """
    if pd.isna(value):
        return 'text-muted'
    elif dtype in ['int64', 'float64']:
        return 'text-end'
    elif dtype == 'datetime64[ns]' or (hasattr(value, '__class__') and value.__class__.__name__ in ['date', 'Timestamp']):
        return 'text-center'
    else:
        return 'text-start'

def generate_preview_html(df, show_index=False):
    """
    データプレビュー用のHTMLを生成
    
    Args:
        df: 表示するDataFrame
        show_index: インデックスを表示するかどうか
        
    Returns:
        生成されたHTML文字列
    """
    html_parts = ['<table class="table table-sm table-striped table-hover">']
    
    # ヘッダー行
    html_parts.append('<thead><tr>')
    if show_index:
        html_parts.append('<th class="text-center">Index</th>')
    for col in df.columns:
        html_parts.append(f'<th class="text-center">{col}</th>')
    html_parts.append('</tr></thead>')
    
    # データ行
    html_parts.append('<tbody>')
    for idx, row in df.head().iterrows():
        html_parts.append('<tr>')
        if show_index:
            cell_class = get_cell_class(idx, type(idx))
            cell_value = str(idx) if pd.notna(idx) else ''
            html_parts.append(f'<td class="{cell_class}">{cell_value}</td>')
        for col in df.columns:
            cell_class = get_cell_class(row[col], df[col].dtype)
            cell_value = str(row[col]) if pd.notna(row[col]) else ''
            html_parts.append(f'<td class="{cell_class}">{cell_value}</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody></table>')
    
    return ''.join(html_parts)

# ===== データ処理関数 =====

def process_index_handling(df, add_index):
    """
    インデックス処理を実行（シンプル版）
    
    Args:
        df: 処理対象のDataFrame
        add_index: インデックス処理の選択肢
        
    Returns:
        処理後のDataFrameと出力インデックス設定
    """
    output_index = False
    
    logger.info(f"インデックス処理開始: add_index='{add_index}'")
    logger.info(f"入力データ形状: {df.shape}, 列名: {df.columns.tolist()}")
    
    if add_index == 'yes':
        # 新たに連番インデックスを付加
        df.insert(0, 'Index', range(1, len(df)+1))
        output_index = False  # 列として追加するので、pandasのインデックスはFalse
        logger.info("新たに連番インデックスを付加しました")
    elif add_index == 'remove':
        # 1列目を削除
        if len(df.columns) > 0:
            first_column = df.columns[0]
            df = df.drop(columns=[first_column])
            output_index = False
            logger.info(f"1列目 '{first_column}' を削除しました")
        else:
            logger.warning("列が存在しないため、削除処理をスキップ")
    elif add_index == 'no':
        # インデックスを付加しない
        output_index = False
        logger.info("インデックスを付加しません")
    else:
        # 予期しない値の場合
        logger.warning(f"予期しないadd_index: {add_index}")
        output_index = False
    
    logger.info(f"最終的な出力インデックス設定: {output_index}")
    logger.info(f"処理後のDataFrame形状: {df.shape}")
    logger.info(f"処理後の列名: {df.columns.tolist()}")
    
    return df, output_index

def process_missing_values(df, missing_strategies, missing_columns):
    """
    欠損値処理を実行
    
    Args:
        df: 処理対象のDataFrame
        missing_strategies: 欠損値処理戦略のリスト
        missing_columns: 処理対象列のリスト
        
    Returns:
        処理後のDataFrame
    """
    for i, col in enumerate(missing_columns):
        if i < len(missing_strategies):
            strategy = missing_strategies[i]
            if strategy == 'drop':
                df = df.dropna(subset=[col])
            elif strategy == 'zero':
                df[col] = df[col].fillna(0)
            elif strategy == 'mean':
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
            elif strategy == 'forward':
                df[col] = df[col].fillna(method='ffill')
            elif strategy == 'backward':
                df[col] = df[col].fillna(method='bfill')
    return df

def process_type_conversions(df, type_conversions, type_columns):
    """
    データ型変換を実行
    
    Args:
        df: 処理対象のDataFrame
        type_conversions: 変換タイプのリスト
        type_columns: 処理対象列のリスト
        
    Returns:
        処理後のDataFrame
    """
    for i, col in enumerate(type_columns):
        if i < len(type_conversions):
            conversion = type_conversions[i]
            if conversion == 'int':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif conversion == 'float':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif conversion == 'string':
                df[col] = df[col].astype(str)
            elif conversion == 'datetime':
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.date
            elif conversion == 'category':
                df[col] = df[col].astype('category')
    return df

def process_summary(df, group_column, agg_column, agg_function):
    """
    集計処理を実行
    
    Args:
        df: 処理対象のDataFrame
        group_column: グループ化列
        agg_column: 集計対象列
        agg_function: 集計関数
        
    Returns:
        集計結果のDataFrame
    """
    agg_functions = {
        'mean': lambda x: x.mean(),
        'sum': lambda x: x.sum(),
        'count': lambda x: x.count(),
        'min': lambda x: x.min(),
        'max': lambda x: x.max(),
        'std': lambda x: x.std(),
        'median': lambda x: x.median()
    }
    
    if agg_function in agg_functions:
        result = df.groupby(group_column)[agg_column].apply(agg_functions[agg_function])
        df = result.to_frame(name=f'{agg_column}_{agg_function}')
        df = df.reset_index()
    
    return df

def process_pivot_table(df, pivot_index, pivot_columns, pivot_values, pivot_aggfunc):
    """
    ピボットテーブルを実行
    
    Args:
        df: 処理対象のDataFrame
        pivot_index: インデックス列
        pivot_columns: 列方向のグループ化列
        pivot_values: 値列
        pivot_aggfunc: 集計関数
        
    Returns:
        ピボットテーブル結果のDataFrame
    """
    pivot_result = df.pivot_table(
        index=pivot_index,
        columns=pivot_columns,
        values=pivot_values,
        aggfunc=pivot_aggfunc
    )
    df = pivot_result.reset_index()
    return df

@csv_processor_bp.route('/csv_processor', methods=['GET', 'POST'])
def processor():
    """CSV加工集計のメインルート"""
    if request.method == 'GET':
        # セッションをクリアして確実に初期化
        session.pop(SESSION_KEY, None)
        context = {
            'form_values': {},
            'filename': None,
            'df_shape': None,
            'df_preview_html': None,
            'processed_filename': None,
            'processed_preview_html': None,
            'columns': [],
            'dtypes': {},
            'missing_info': {},
            'missing_percentage': {},
            'numeric_columns': [],
            'categorical_columns': [],
            'basic_stats': {},
            'has_index_column': False,
            'index_column_name': None
        }
        
        return render_template('csv_processor.html', **context)

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
        elif action == 'process_data':
            context = handle_data_processing(context)
        elif action == 'download':
            return handle_download(context)
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
            
            flash('CSV加工データがリセットされました。', 'info')
            return redirect(url_for('csv_processor.processor'))
        else:
            flash('無効なアクションです。', 'warning')

    except Exception as e:
        error_message = ErrorHandler.handle_model_error(e)
        flash(f'エラーが発生しました: {error_message}', 'danger')
        logger.error(f"CSV加工集計でエラーが発生しました: {e}")

    # コンテキストをセッションに保存
    session[SESSION_KEY] = context
    
    # POST処理後のリダイレクト時は、コンテキストを直接渡す
    if request.method == 'POST':
        return render_template('csv_processor.html', **context)
    
    return redirect(url_for('csv_processor.processor'))

def handle_file_upload(context):
    """ファイルアップロードを処理"""
    try:
        file = request.files.get('file')
        
        if not file:
            flash('ファイルが選択されていません。', 'warning')
            return context
            
        upload_path = FileManager.save_uploaded_file(
            file, 
            current_app.config['UPLOAD_FOLDER']
        )
        
        filename = os.path.basename(upload_path)
        context = {'filename': filename, 'form_values': {}}
        
        # CSVファイルの読み込み
        df = DataProcessor.load_csv_safe(upload_path, index_col=None, auto_index_detection=False)
        logger.info(f"データ読み込み完了: 形状={df.shape}, 列名={df.columns.tolist()}")
        logger.info(f"読み込み後のインデックス: {df.index.tolist()[:5]}...")
        logger.info(f"読み込み後のインデックス名: {df.index.name}")
        
        # Unnamed: 0列を検出してブランクに変更
        if len(df.columns) > 0 and df.columns[0] == 'Unnamed: 0':
            logger.info("Unnamed: 0列を検出しました。ブランクに変更します。")
            df.columns.values[0] = ''
        
        # 最初の列が空かどうかをチェック
        if len(df.columns) > 0:
            first_col = df.columns[0]
            first_col_empty = df[first_col].isna().all() or (df[first_col] == '').all()
            logger.info(f"最初の列 '{first_col}' が空かどうか: {first_col_empty}")
            if first_col_empty:
                logger.info("最初の列が空のため、インデックス列として扱います")
                # 最初の列を削除してインデックスとして設定
                df = df.drop(columns=[first_col])
                df.index = range(1, len(df)+1)
                df.index.name = 'Index'
                logger.info(f"インデックス設定後: 形状={df.shape}, 列名={df.columns.tolist()}")
        
        # データの基本情報を設定
        context['columns'] = df.columns.tolist()
        context['df_shape'] = df.shape
        
        # プレビュー用にUnnamed: 0列をブランクに変更
        preview_df = df.copy()
        if len(preview_df.columns) > 0 and preview_df.columns[0] == 'Unnamed: 0':
            logger.info("初期プレビュー用にUnnamed: 0列をブランクに変更します。")
            preview_df.columns.values[0] = ''
        
        context['df_preview_html'] = generate_preview_html(preview_df, show_index=False)
        
        # データ型情報
        context['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
        
        # 欠損値情報
        missing_info = df.isnull().sum()
        context['missing_info'] = missing_info.to_dict()
        context['missing_percentage'] = (missing_info / len(df) * 100).to_dict()
        
        # 数値列とカテゴリ列の分類
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        context['numeric_columns'] = numeric_columns
        context['categorical_columns'] = categorical_columns
        
        # 基本統計情報
        if len(numeric_columns) > 0:
            context['basic_stats'] = df[numeric_columns].describe().to_dict()
        
        flash(f'ファイル "{filename}" が正常にアップロードされました。', 'success')
        logger.info(f"CSVファイルアップロード完了: {filename}")
        
        return context
        
    except Exception as e:
        error_message = ErrorHandler.handle_upload_error(e)
        flash(error_message, 'warning')
        raise

def handle_data_processing(context):
    """データ加工処理を実行"""
    try:
        logger.info("=== データ加工処理開始 ===")
        
        # ファイルがアップロードされているかチェック
        if not context.get('filename'):
            flash('ファイルを選択してください。', 'warning')
            return context
        
        # ファイルの存在確認
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], context['filename'])
        if not os.path.exists(file_path):
            flash('アップロードされたファイルが見つかりません。', 'warning')
            return context
        
        logger.info(f"処理対象ファイル: {file_path}")
        
        # 元データの読み込み
        df = DataProcessor.load_csv_safe(file_path, index_col=None, auto_index_detection=False)
        logger.info(f"データ読み込み完了: 形状={df.shape}, 列名={df.columns.tolist()}")
        logger.info(f"読み込み後のインデックス: {df.index.tolist()[:5]}...")
        logger.info(f"読み込み後のインデックス名: {df.index.name}")
        
        # Unnamed: 0列を検出してブランクに変更
        if len(df.columns) > 0 and df.columns[0] == 'Unnamed: 0':
            logger.info("Unnamed: 0列を検出しました。ブランクに変更します。")
            df.columns.values[0] = ''
        
        # 最初の列が空かどうかをチェック
        if len(df.columns) > 0:
            first_col = df.columns[0]
            first_col_empty = df[first_col].isna().all() or (df[first_col] == '').all()
            logger.info(f"最初の列 '{first_col}' が空かどうか: {first_col_empty}")
            if first_col_empty:
                logger.info("最初の列が空のため、インデックス列として扱います")
                # 最初の列を削除してインデックスとして設定
                df = df.drop(columns=[first_col])
                df.index = range(1, len(df)+1)
                df.index.name = 'Index'
                logger.info(f"インデックス設定後: 形状={df.shape}, 列名={df.columns.tolist()}")
        
        # データの基本情報を設定
        context['columns'] = df.columns.tolist()
        context['df_shape'] = df.shape
        
        # プレビュー用にUnnamed: 0列をブランクに変更
        preview_df = df.copy()
        if len(preview_df.columns) > 0 and preview_df.columns[0] == 'Unnamed: 0':
            logger.info("初期プレビュー用にUnnamed: 0列をブランクに変更します。")
            preview_df.columns.values[0] = ''
        
        context['df_preview_html'] = generate_preview_html(preview_df, show_index=False)
        
        # データ型情報
        context['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
        
        # 欠損値情報
        missing_info = df.isnull().sum()
        context['missing_info'] = missing_info.to_dict()
        context['missing_percentage'] = (missing_info / len(df) * 100).to_dict()
        
        # 数値列とカテゴリ列の分類
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        context['numeric_columns'] = numeric_columns
        context['categorical_columns'] = categorical_columns
        
        # 基本統計情報
        if len(numeric_columns) > 0:
            context['basic_stats'] = df[numeric_columns].describe().to_dict()
        
        # インデックス列の判定結果をコンテキストに追加
        context['has_index_column'] = df.index.name is not None or not df.index.equals(pd.RangeIndex(len(df)))
        if context['has_index_column'] and len(df.columns) > 0:
            context['index_column_name'] = df.columns[0]
        
        # インデックス処理
        logger.info("=== インデックス処理開始 ===")
        add_index = request.form.get('add_index', None)
        
        logger.info(f"ユーザー選択: add_index='{add_index}'")
        
        df, output_index = process_index_handling(df, add_index)
        logger.info(f"インデックス処理完了: output_index={output_index}")
        
        # 欠損値処理
        missing_strategies = request.form.getlist('missing_strategies')
        missing_columns = request.form.getlist('missing_columns')
        df = process_missing_values(df, missing_strategies, missing_columns)
        
        # データ型変換
        type_conversions = request.form.getlist('type_conversions')
        type_columns = request.form.getlist('type_columns')
        df = process_type_conversions(df, type_conversions, type_columns)
        
        # フィルタリング
        filter_condition = request.form.get('filter_condition', '').strip()
        if filter_condition:
            logger.info(f"フィルタリング条件: {filter_condition}")
            try:
                df = df.query(filter_condition)
                logger.info(f"フィルタリング後の行数: {len(df)}")
            except Exception as e:
                logger.error(f"フィルタリングエラー: {e}")
                flash(f'フィルタリング条件が正しくありません: {filter_condition}', 'warning')
        
        # 重複行削除
        if request.form.get('remove_duplicates'):
            logger.info("重複行削除を実行")
            original_count = len(df)
            df = df.drop_duplicates()
            removed_count = original_count - len(df)
            logger.info(f"重複行削除完了: {removed_count}行削除")
        
        # ソート
        sort_column = request.form.get('sort_column', '').strip()
        if sort_column and sort_column in df.columns:
            sort_ascending = request.form.get('sort_ascending', 'asc') == 'asc'
            logger.info(f"ソート実行: {sort_column}, 昇順: {sort_ascending}")
            df = df.sort_values(by=sort_column, ascending=sort_ascending)
        
        # 集計処理
        enable_summary = request.form.get('enable_summary') == '1'
        if enable_summary:
            group_column = request.form.get('group_columns', '').strip()
            agg_column = request.form.get('agg_columns', '').strip()
            agg_function = request.form.get('agg_functions', '').strip()
            
            if group_column and agg_column and agg_function:
                logger.info(f"集計処理: {group_column}でグループ化, {agg_column}を{agg_function}")
                try:
                    df = process_summary(df, group_column, agg_column, agg_function)
                    logger.info("集計処理完了")
                except Exception as e:
                    logger.error(f"集計処理エラー: {e}")
                    flash('集計処理でエラーが発生しました。', 'warning')
        
        # ピボットテーブル
        enable_pivot = request.form.get('enable_pivot') == '1'
        if enable_pivot:
            pivot_index = request.form.get('pivot_index', '').strip()
            pivot_columns = request.form.get('pivot_columns', '').strip()
            pivot_values = request.form.get('pivot_values', '').strip()
            pivot_aggfunc = request.form.get('pivot_aggfunc', '').strip()
            
            if pivot_index and pivot_columns and pivot_values and pivot_aggfunc:
                logger.info(f"ピボットテーブル作成: {pivot_index}, {pivot_columns}, {pivot_values}, {pivot_aggfunc}")
                try:
                    df = process_pivot_table(df, pivot_index, pivot_columns, pivot_values, pivot_aggfunc)
                    logger.info("ピボットテーブル作成完了")
                except Exception as e:
                    logger.error(f"ピボットテーブルエラー: {e}")
                    flash('ピボットテーブルの作成でエラーが発生しました。', 'warning')
        
        # 処理結果の保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_filename = f"processed_{timestamp}_{context['filename']}"
        processed_path = os.path.join(current_app.config['UPLOAD_FOLDER'], processed_filename)
        
        # 保存前にUnnamed: 0列をブランクに変更
        if len(df.columns) > 0 and df.columns[0] == 'Unnamed: 0':
            logger.info("保存前にUnnamed: 0列をブランクに変更します。")
            df.columns.values[0] = ''
        
        logger.info(f"CSV保存: 形状={df.shape}, 列名={df.columns.tolist()}, index={output_index}")
        df.to_csv(processed_path, index=output_index)
        
        context['processed_filename'] = processed_filename
        context['processed_shape'] = df.shape
        context['output_index'] = output_index
        
        # 処理結果のプレビュー生成
        try:
            # プレビュー用にUnnamed: 0列をブランクに変更
            preview_df = df.copy()
            if len(preview_df.columns) > 0 and preview_df.columns[0] == 'Unnamed: 0':
                logger.info("プレビュー用にUnnamed: 0列をブランクに変更します。")
                preview_df.columns.values[0] = ''
            
            context['processed_preview_html'] = generate_preview_html(preview_df, show_index=output_index)
            logger.info(f"プレビュー生成完了: 行数={len(df.head())}, インデックス表示={output_index}")
        except Exception as e:
            logger.error(f"プレビュー生成でエラーが発生しました: {e}")
            context['processed_preview_html'] = f'<div class="alert alert-danger">プレビュー生成でエラーが発生しました: {e}</div>'
        
        flash('データの加工が完了しました。', 'success')
        logger.info(f"データ加工完了: {processed_filename}")
        
        return context
        
    except Exception as e:
        logger.error(f"データ加工エラー: {e}")
        flash('データ加工中にエラーが発生しました。', 'warning')
        raise

def handle_download(context):
    """処理結果のダウンロードを処理"""
    try:
        filename = context.get('processed_filename')
        if not filename:
            flash('ダウンロードするファイルがありません。', 'warning')
            return redirect(url_for('csv_processor.processor'))
        
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            flash('ファイルが見つかりません。', 'warning')
            return redirect(url_for('csv_processor.processor'))
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"ダウンロードエラー: {e}")
        flash('ダウンロード中にエラーが発生しました。', 'warning')
        return redirect(url_for('csv_processor.processor'))

@csv_processor_bp.route('/csv_processor_datasets')
def datasets():
    """CSV加工集計用サンプルデータセットページ"""
    return render_template('csv_processor_datasets.html')

@csv_processor_bp.route('/api/column_stats/<filename>')
def column_stats(filename):
    """列の統計情報を取得するAPI"""
    try:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        df = DataProcessor.load_csv_safe(file_path)
        
        stats = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    'type': 'numeric',
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'missing': int(df[col].isnull().sum())
                }
            else:
                stats[col] = {
                    'type': 'categorical',
                    'unique_count': int(df[col].nunique()),
                    'missing': int(df[col].isnull().sum())
                }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"列統計取得エラー: {e}")
        return jsonify({'error': str(e)}), 500
