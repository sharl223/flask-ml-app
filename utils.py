"""
AI Playground 共通ユーティリティモジュール
データ処理、ファイル操作、エラーハンドリングなどの共通機能を提供
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import base64
import io
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, r2_score
from flask import current_app, flash
import warnings

# 警告を無視
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """ログ設定を初期化する"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ai_playground.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

class DataProcessor:
    """データ処理の共通クラス"""
    
    @staticmethod
    def load_csv_safe(file_path: str, index_col: Optional[int] = None, auto_index_detection: bool = True) -> pd.DataFrame:
        """
        安全にCSVファイルを読み込む
        
        Args:
            file_path: CSVファイルのパス
            index_col: インデックス列の位置（Noneの場合はインデックス列なし）
            auto_index_detection: 自動インデックス検出を有効にするかどうか
            
        Returns:
            読み込まれたDataFrame
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            pd.errors.EmptyDataError: ファイルが空の場合
            pd.errors.ParserError: パースエラーの場合
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
            
            # まず通常のCSVとして読み込み
            df = pd.read_csv(file_path, index_col=index_col)
            
            if df.empty:
                raise pd.errors.EmptyDataError("ファイルが空です")
            
            # インデックス列が指定されていない場合、最初の列が連番の場合はインデックスとして設定
            if auto_index_detection and index_col is None and len(df.columns) > 0:
                first_col = df.columns[0]
                # 最初の列が数値で連番の場合はインデックスとして設定
                if (df[first_col].dtype in ['int64', 'float64'] and 
                    df[first_col].is_monotonic_increasing and 
                    df[first_col].iloc[0] == 0 or df[first_col].iloc[0] == 1):
                    logger.info(f"最初の列 '{first_col}' をインデックスとして設定")
                    df = df.set_index(first_col)
                
            logger.info(f"CSVファイルを正常に読み込みました: {file_path}, 形状: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"CSVファイルの読み込みに失敗しました: {file_path}, エラー: {e}")
            raise
    
    @staticmethod
    def detect_categorical_features(df: pd.DataFrame, max_unique: int = 20) -> List[str]:
        """
        カテゴリカル特徴量を自動検出
        
        Args:
            df: 対象のDataFrame
            max_unique: カテゴリカルとみなす最大ユニーク値数
            
        Returns:
            カテゴリカル特徴量のリスト
        """
        categorical_features = []
        
        for col in df.columns:
            # オブジェクト型またはブール型
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                categorical_features.append(col)
            # 数値型だがユニーク値が少ない場合
            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < max_unique:
                series_no_na = df[col].dropna()
                if not series_no_na.empty and series_no_na.isin([0, 1]).all():
                    categorical_features.append(col)
        
        logger.info(f"カテゴリカル特徴量を検出しました: {categorical_features}")
        return categorical_features
    
    @staticmethod
    def prepare_data_for_lgbm(
        df: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        categorical_features: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
        """
        LightGBM用のデータ前処理
        
        Args:
            df: 入力DataFrame
            target_column: ターゲット列名
            feature_columns: 特徴量列名のリスト
            categorical_features: カテゴリカル特徴量のリスト
            
        Returns:
            前処理済み特徴量、ターゲット、ラベルエンコーダーの辞書
        """
        try:
            df_processed = df[feature_columns].copy()
            label_encoders = {}
            
            # カテゴリカル特徴量のラベルエンコーディング
            for col in categorical_features:
                if col in df_processed.columns:
                    le = LabelEncoder()
                    valid_data = df_processed[col][df_processed[col].notna()]
                    
                    if len(valid_data) > 0:
                        le.fit(valid_data)
                        df_processed[col] = df_processed[col].map(
                            lambda s: le.transform([s])[0] if pd.notna(s) and s in le.classes_ else s
                        )
                        label_encoders[col] = le
            
            X = df_processed
            y = df[target_column] if target_column in df.columns else None
            
            logger.info(f"LightGBM用データ前処理完了: 特徴量形状={X.shape}, ターゲット形状={y.shape if y is not None else 'None'}")
            return X, y, label_encoders
            
        except Exception as e:
            logger.error(f"LightGBM用データ前処理に失敗しました: {e}")
            raise
    
    @staticmethod
    def prepare_data_for_linear(
        df: pd.DataFrame, 
        feature_columns: List[str], 
        numeric_categoricals: List[str], 
        scaler: Optional[StandardScaler] = None, 
        train_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
        """
        線形モデル用のデータ前処理
        
        Args:
            df: 入力DataFrame
            feature_columns: 特徴量列名のリスト
            numeric_categoricals: 数値カテゴリカル特徴量のリスト
            scaler: 既存のスケーラー（予測時）
            train_columns: 学習時の列名（予測時）
            
        Returns:
            前処理済み特徴量、スケーラー、列名リスト
        """
        try:
            X = df[feature_columns].copy()
            
            # 指定された数値列をカテゴリ型に変換
            for col in numeric_categoricals:
                if col in X.columns:
                    X[col] = X[col].astype('category')
            
            # ダミー変数化
            X = pd.get_dummies(X, drop_first=True, dtype=float)
            
            if train_columns is not None:
                # 予測時：学習時と列を揃える
                missing_cols = set(train_columns) - set(X.columns)
                for c in missing_cols:
                    X[c] = 0
                X = X[train_columns]
            else:
                # 学習時
                train_columns = X.columns.tolist()
            
            # 数値列の標準化
            numeric_cols = X.select_dtypes(include=['number']).columns
            
            if scaler is None:
                # 学習時
                scaler = StandardScaler()
                if len(numeric_cols) > 0:
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            else:
                # 予測時
                if len(numeric_cols) > 0:
                    X[numeric_cols] = scaler.transform(X[numeric_cols])
            
            logger.info(f"線形モデル用データ前処理完了: 形状={X.shape}")
            return X, scaler, train_columns
            
        except Exception as e:
            logger.error(f"線形モデル用データ前処理に失敗しました: {e}")
            raise

class ModelManager:
    """モデル管理の共通クラス"""
    
    @staticmethod
    def save_model(
        model_data: Dict[str, Any], 
        filename: str, 
        models_folder: str
    ) -> str:
        """
        モデルを安全に保存
        
        Args:
            model_data: 保存するモデルデータ
            filename: ファイル名
            models_folder: 保存先フォルダ
            
        Returns:
            保存されたファイルのパス
        """
        try:
            os.makedirs(models_folder, exist_ok=True)
            model_path = os.path.join(models_folder, filename)
            
            joblib.dump(model_data, model_path)
            logger.info(f"モデルを保存しました: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"モデルの保存に失敗しました: {e}")
            raise
    
    @staticmethod
    def load_model(model_path: str) -> Dict[str, Any]:
        """
        モデルを安全に読み込み
        
        Args:
            model_path: モデルファイルのパス
            
        Returns:
            読み込まれたモデルデータ
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
            model_data = joblib.load(model_path)
            logger.info(f"モデルを読み込みました: {model_path}")
            return model_data
            
        except Exception as e:
            logger.error(f"モデルの読み込みに失敗しました: {e}")
            raise

class VisualizationHelper:
    """可視化の共通クラス"""
    
    @staticmethod
    def create_feature_importance_plot(
        model, 
        feature_names: List[str], 
        title: str = "予測に重要な影響を与えた情報 Top 20"
    ) -> str:
        """
        特徴量重要度のプロットを生成
        
        Args:
            model: 学習済みモデル
            feature_names: 特徴量名のリスト
            title: プロットのタイトル
            
        Returns:
            base64エンコードされた画像データ
        """
        try:
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names, 
                'importance': importance
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title(title, fontsize=18)
            plt.xlabel('重要度', fontsize=14)
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=20)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            logger.info("特徴量重要度プロットを生成しました")
            return img_str
            
        except Exception as e:
            logger.error(f"特徴量重要度プロットの生成に失敗しました: {e}")
            raise
    
    @staticmethod
    def create_coefficient_plot(
        coeffs: pd.DataFrame, 
        title: str = "予測への影響度（係数）"
    ) -> str:
        """
        係数のプロットを生成
        
        Args:
            coeffs: 係数のDataFrame
            title: プロットのタイトル
            
        Returns:
            base64エンコードされた画像データ
        """
        try:
            top_coeffs = pd.concat([coeffs.head(10), coeffs.tail(10)]).sort_values('係数', ascending=True)
            
            plt.figure(figsize=(10, 8))
            plt.barh(
                top_coeffs.index, 
                top_coeffs['係数'], 
                color=top_coeffs['係数'].map(lambda x: 'tomato' if x > 0 else 'dodgerblue')
            )
            plt.title(title, fontsize=18)
            plt.xlabel('係数の大きさ', fontsize=14)
            plt.tick_params(axis='both', labelsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            logger.info("係数プロットを生成しました")
            return img_str
            
        except Exception as e:
            logger.error(f"係数プロットの生成に失敗しました: {e}")
            raise

class FileManager:
    """ファイル操作の共通クラス"""
    
    @staticmethod
    def save_uploaded_file(file, upload_folder: str) -> str:
        """
        アップロードされたファイルを安全に保存
        
        Args:
            file: アップロードされたファイル
            upload_folder: 保存先フォルダ
            
        Returns:
            保存されたファイルのパス
        """
        try:
            os.makedirs(upload_folder, exist_ok=True)
            
            if not file or file.filename == '':
                raise ValueError("ファイルが選択されていません")
            
            filename = file.filename
            upload_path = os.path.join(upload_folder, filename)
            
            file.save(upload_path)
            logger.info(f"ファイルをアップロードしました: {upload_path}")
            return upload_path
            
        except Exception as e:
            logger.error(f"ファイルのアップロードに失敗しました: {e}")
            raise
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        ファイルの基本情報を取得
        
        Args:
            file_path: ファイルパス
            
        Returns:
            ファイル情報の辞書
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
            
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'exists': True
            }
            
        except Exception as e:
            logger.error(f"ファイル情報の取得に失敗しました: {e}")
            return {'exists': False, 'error': str(e)}

class ValidationHelper:
    """バリデーションの共通クラス"""
    
    @staticmethod
    def validate_csv_file(file_path: str) -> bool:
        """
        CSVファイルの妥当性を検証
        
        Args:
            file_path: CSVファイルのパス
            
        Returns:
            妥当性の結果
        """
        try:
            df = DataProcessor.load_csv_safe(file_path)
            
            # 基本的な検証
            if df.empty:
                return False
            
            if df.shape[0] < 10:  # 最低10行必要
                return False
            
            if df.shape[1] < 2:   # 最低2列必要
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"CSVファイルの検証に失敗しました: {e}")
            return False
    
    @staticmethod
    def validate_model_parameters(params: Dict[str, Any]) -> bool:
        """
        モデルパラメータの妥当性を検証
        
        Args:
            params: パラメータの辞書
            
        Returns:
            妥当性の結果
        """
        try:
            # 基本的なパラメータチェック
            required_params = ['learning_rate', 'n_estimators', 'max_depth']
            
            for param in required_params:
                if param not in params:
                    return False
                
                value = params[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"パラメータの検証に失敗しました: {e}")
            return False

class ErrorHandler:
    """エラーハンドリングの共通クラス"""
    
    @staticmethod
    def handle_upload_error(error: Exception) -> str:
        """
        アップロードエラーを処理
        
        Args:
            error: 発生したエラー
            
        Returns:
            ユーザーフレンドリーなエラーメッセージ
        """
        error_type = type(error).__name__
        
        if isinstance(error, FileNotFoundError):
            return "ファイルが見つかりません。"
        elif isinstance(error, pd.errors.EmptyDataError):
            return "ファイルが空です。"
        elif isinstance(error, pd.errors.ParserError):
            return "ファイルの形式が正しくありません。CSVファイルを確認してください。"
        elif isinstance(error, ValueError):
            return str(error)
        else:
            logger.error(f"予期しないエラーが発生しました: {error}")
            return "予期しないエラーが発生しました。もう一度お試しください。"
    
    @staticmethod
    def handle_model_error(error: Exception) -> str:
        """
        モデル関連エラーを処理
        
        Args:
            error: 発生したエラー
            
        Returns:
            ユーザーフレンドリーなエラーメッセージ
        """
        error_type = type(error).__name__
        
        if isinstance(error, ValueError):
            return "データの形式が正しくありません。"
        elif isinstance(error, KeyError):
            return "必要な列が見つかりません。"
        elif isinstance(error, MemoryError):
            return "メモリが不足しています。データサイズを小さくしてください。"
        else:
            logger.error(f"モデル処理で予期しないエラーが発生しました: {error}")
            return "モデルの処理中にエラーが発生しました。もう一度お試しください。"

# 便利な関数
def safe_float(value: Any, default: float = 0.0) -> float:
    """安全にfloatに変換"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """安全にintに変換"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def generate_unique_filename(prefix: str, extension: str) -> str:
    """ユニークなファイル名を生成"""
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:8]}.{extension}"

def cleanup_temp_files(folder: str, pattern: str = "*.tmp") -> int:
    """
    一時ファイルをクリーンアップ
    
    Args:
        folder: クリーンアップ対象フォルダ
        pattern: ファイルパターン
        
    Returns:
        削除されたファイル数
    """
    try:
        import glob
        files = glob.glob(os.path.join(folder, pattern))
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"一時ファイルの削除に失敗: {file}, エラー: {e}")
        return len(files)
    except Exception as e:
        logger.error(f"一時ファイルクリーンアップでエラー: {e}")
        return 0

def clear_all_sessions(session):
    """
    全セッション情報をクリア
    
    Args:
        session: Flaskセッションオブジェクト
        
    Returns:
        クリアされたセッションキーのリスト
    """
    try:
        # 全セッションキーの定義
        session_keys = [
            'csv_processor_context',
            'linear_regression_context', 
            'logistic_regression_context',
            'lgbm_playground_context',
            'lgbm'
        ]
        
        cleared_keys = []
        
        # 各セッションキーをクリア
        for key in session_keys:
            if key in session:
                session.pop(key, None)
                cleared_keys.append(key)
                logger.info(f"セッションキー '{key}' をクリアしました")
        
        # その他のセッション情報もクリア
        session.clear()
        
        # セッションを確実にクリアするため、空の辞書を設定
        session.modified = True
        
        # セッションファイルも削除を試行
        try:
            from flask import current_app
            if current_app and 'SESSION_FILE_DIR' in current_app.config:
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
        
        logger.info(f"全セッション情報をクリアしました。クリアされたキー: {cleared_keys}")
        return cleared_keys
        
    except Exception as e:
        logger.error(f"セッションクリアでエラーが発生しました: {e}")
        return [] 