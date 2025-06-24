"""
CSV加工編集ページ用サンプルデータセット作成スクリプト
scikit-learnとseabornから実用的なデータセットを生成
"""

import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, 
    make_classification, make_regression, fetch_california_housing
)
import seaborn as sns
from datetime import datetime, timedelta
import random
import os

def create_iris_dataset():
    """アイリスデータセット（分類問題）"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    # 欠損値を追加（加工練習用）
    df.loc[random.sample(range(len(df)), 10), 'sepal length (cm)'] = np.nan
    df.loc[random.sample(range(len(df)), 5), 'sepal width (cm)'] = np.nan
    
    # インデックス列を追加
    df['id'] = range(1, len(df) + 1)
    df = df.set_index('id')
    
    return df, "アイリス花の分類データ（欠損値あり）"

def create_wine_dataset():
    """ワインデータセット（分類問題）"""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    df['wine_type'] = [wine.target_names[i] for i in wine.target]
    
    # 欠損値を追加
    df.loc[random.sample(range(len(df)), 8), 'alcohol'] = np.nan
    df.loc[random.sample(range(len(df)), 6), 'malic_acid'] = np.nan
    
    # 重複行を追加（加工練習用）
    duplicate_rows = df.sample(n=5)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # インデックス列を追加
    df['id'] = range(1, len(df) + 1)
    df = df.set_index('id')
    
    return df, "ワインの品質分類データ（重複行あり）"

def create_breast_cancer_dataset():
    """乳がんデータセット（分類問題）"""
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    df['diagnosis'] = ['malignant' if i == 0 else 'benign' for i in cancer.target]
    
    # 欠損値を追加
    df.loc[random.sample(range(len(df)), 12), 'mean radius'] = np.nan
    df.loc[random.sample(range(len(df)), 8), 'mean texture'] = np.nan
    
    # インデックス列を追加
    df['id'] = range(1, len(df) + 1)
    df = df.set_index('id')
    
    return df, "乳がん診断データ（欠損値あり）"

def create_sales_dataset():
    """売上データセット（回帰問題）"""
    np.random.seed(42)
    n_samples = 1000
    
    # 基本データ
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    regions = ['東京', '大阪', '名古屋', '福岡', '札幌', '仙台', '広島', '金沢']
    products = ['A商品', 'B商品', 'C商品', 'D商品', 'E商品']
    
    data = {
        'date': dates,
        'region': np.random.choice(regions, n_samples),
        'product': np.random.choice(products, n_samples),
        'quantity': np.random.poisson(50, n_samples),
        'unit_price': np.random.normal(1000, 200, n_samples),
        'customer_age': np.random.normal(45, 15, n_samples),
        'customer_gender': np.random.choice(['男性', '女性'], n_samples),
        'discount_rate': np.random.uniform(0, 0.3, n_samples),
        'weather': np.random.choice(['晴れ', '曇り', '雨', '雪'], n_samples),
        'holiday': np.random.choice([True, False], n_samples, p=[0.1, 0.9])
    }
    
    df = pd.DataFrame(data)
    
    # 売上計算
    df['sales'] = df['quantity'] * df['unit_price'] * (1 - df['discount_rate'])
    
    # 欠損値を追加
    df.loc[random.sample(range(len(df)), 50), 'quantity'] = np.nan
    df.loc[random.sample(range(len(df)), 30), 'unit_price'] = np.nan
    df.loc[random.sample(range(len(df)), 20), 'customer_age'] = np.nan
    
    # 異常値を追加
    df.loc[random.sample(range(len(df)), 5), 'quantity'] = 9999
    df.loc[random.sample(range(len(df)), 3), 'unit_price'] = -100
    
    # インデックス列を追加
    df['id'] = range(1, len(df) + 1)
    df = df.set_index('id')
    
    return df, "売上分析データ（欠損値・異常値あり）"

def create_employee_dataset():
    """従業員データセット（分類・回帰問題）"""
    np.random.seed(42)
    n_samples = 500
    
    departments = ['営業', '開発', '人事', '経理', 'マーケティング', 'サポート']
    positions = ['一般', '主任', '課長', '部長', '取締役']
    education = ['高校', '専門学校', '短大', '大学', '大学院']
    
    data = {
        'employee_id': range(1, n_samples + 1),
        'name': [f'従業員{i:03d}' for i in range(1, n_samples + 1)],
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'gender': np.random.choice(['男性', '女性'], n_samples),
        'department': np.random.choice(departments, n_samples),
        'position': np.random.choice(positions, n_samples),
        'education': np.random.choice(education, n_samples),
        'years_of_service': np.random.exponential(5, n_samples).astype(int),
        'salary': np.random.normal(400000, 150000, n_samples),
        'performance_score': np.random.normal(75, 15, n_samples),
        'satisfaction_score': np.random.uniform(1, 5, n_samples),
        'overtime_hours': np.random.exponential(20, n_samples),
        'projects_completed': np.random.poisson(8, n_samples),
        'training_hours': np.random.exponential(40, n_samples),
        'promoted': np.random.choice([True, False], n_samples, p=[0.2, 0.8])
    }
    
    df = pd.DataFrame(data)
    
    # 欠損値を追加
    df.loc[random.sample(range(len(df)), 30), 'salary'] = np.nan
    df.loc[random.sample(range(len(df)), 20), 'performance_score'] = np.nan
    df.loc[random.sample(range(len(df)), 15), 'satisfaction_score'] = np.nan
    
    # インデックス列を追加
    df['id'] = range(1, len(df) + 1)
    df = df.set_index('id')
    
    return df, "従業員分析データ（欠損値あり）"

def create_customer_dataset():
    """顧客データセット（分類問題）"""
    np.random.seed(42)
    n_samples = 800
    
    # 基本データ
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'gender': np.random.choice(['男性', '女性'], n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'education': np.random.choice(['高校', '専門学校', '短大', '大学', '大学院'], n_samples),
        'marital_status': np.random.choice(['独身', '既婚', '離婚'], n_samples),
        'children': np.random.poisson(1, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'purchase_frequency': np.random.exponential(5, n_samples),
        'avg_purchase_amount': np.random.lognormal(8, 0.3, n_samples),
        'days_since_last_purchase': np.random.exponential(30, n_samples),
        'total_purchases': np.random.poisson(20, n_samples),
        'returns_count': np.random.poisson(2, n_samples),
        'satisfaction_score': np.random.uniform(1, 5, n_samples),
        'churned': np.random.choice([True, False], n_samples, p=[0.15, 0.85])
    }
    
    df = pd.DataFrame(data)
    
    # 欠損値を追加
    df.loc[random.sample(range(len(df)), 40), 'income'] = np.nan
    df.loc[random.sample(range(len(df)), 25), 'credit_score'] = np.nan
    df.loc[random.sample(range(len(df)), 20), 'satisfaction_score'] = np.nan
    
    # インデックス列を追加
    df['id'] = range(1, len(df) + 1)
    df = df.set_index('id')
    
    return df, "顧客分析データ（欠損値あり）"

def create_weather_dataset():
    """天気データセット（時系列・回帰問題）"""
    np.random.seed(42)
    n_samples = 365
    
    # 基本データ
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # 季節性を考慮したデータ生成
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    seasonal_humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365 + np.pi/4)
    
    data = {
        'date': dates,
        'temperature': seasonal_temp + np.random.normal(0, 3, n_samples),
        'humidity': np.clip(seasonal_humidity + np.random.normal(0, 10, n_samples), 0, 100),
        'pressure': np.random.normal(1013, 10, n_samples),
        'wind_speed': np.random.exponential(5, n_samples),
        'precipitation': np.random.exponential(2, n_samples),
        'visibility': np.random.uniform(5, 20, n_samples),
        'uv_index': np.random.uniform(0, 10, n_samples),
        'weather_condition': np.random.choice(['晴れ', '曇り', '雨', '雪', '霧'], n_samples, p=[0.4, 0.3, 0.2, 0.05, 0.05]),
        'holiday': np.random.choice([True, False], n_samples, p=[0.1, 0.9])
    }
    
    df = pd.DataFrame(data)
    
    # 欠損値を追加
    df.loc[random.sample(range(len(df)), 30), 'temperature'] = np.nan
    df.loc[random.sample(range(len(df)), 25), 'humidity'] = np.nan
    df.loc[random.sample(range(len(df)), 15), 'pressure'] = np.nan
    
    # インデックス列を追加
    df['id'] = range(1, len(df) + 1)
    df = df.set_index('id')
    
    return df, "天気予測データ（欠損値あり）"

def main():
    """メイン処理"""
    print("CSV加工編集ページ用サンプルデータセットを作成中...")
    
    # データセット作成
    datasets = [
        create_iris_dataset(),
        create_wine_dataset(),
        create_breast_cancer_dataset(),
        create_sales_dataset(),
        create_employee_dataset(),
        create_customer_dataset(),
        create_weather_dataset()
    ]
    
    # ファイル保存
    for i, (df, description) in enumerate(datasets, 1):
        filename = f"csv_processor_dataset_{i:02d}.csv"
        filepath = os.path.join('uploads', filename)
        
        # ファイル保存
        df.to_csv(filepath, encoding='utf-8')
        
        print(f"✓ {filename} を作成しました")
        print(f"  説明: {description}")
        print(f"  形状: {df.shape}")
        print(f"  欠損値: {df.isnull().sum().sum()}個")
        print()
    
    print("すべてのデータセットの作成が完了しました！")
    print("\n作成されたデータセット:")
    print("1. アイリス花の分類データ（欠損値あり）")
    print("2. ワインの品質分類データ（重複行あり）")
    print("3. 乳がん診断データ（欠損値あり）")
    print("4. 売上分析データ（欠損値・異常値あり）")
    print("5. 従業員分析データ（欠損値あり）")
    print("6. 顧客分析データ（欠損値あり）")
    print("7. 天気予測データ（欠損値あり）")

if __name__ == "__main__":
    main() 