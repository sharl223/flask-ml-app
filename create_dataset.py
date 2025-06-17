import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

print("カリフォルニア住宅価格データセットをダウンロード・処理しています...")

# 1. データの読み込み
housing = fetch_california_housing()

# 2. データフレームの作成
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target # 目的変数（住宅価格の中央値）を追加

# 3. データの分割
# データを学習用(train)と予測用(predict)に8:2で分割
train_df, predict_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# 4. CSVファイルとして保存
# アプリケーションで扱いやすいように、インデックスをリセットして連番のインデックス列を付与
train_df.reset_index(drop=True).to_csv('california_train.csv', index=True)
predict_df.reset_index(drop=True).to_csv('california_predict.csv', index=True)

print("california_train.csv と california_predict.csv を作成しました。")