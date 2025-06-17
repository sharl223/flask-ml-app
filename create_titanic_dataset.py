import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

print("タイタニックのデータセットをダウンロード・処理しています...")

# 1. データの読み込み
df = sns.load_dataset('titanic')

# 2. 簡単な前処理
df = df.drop(columns=['who', 'adult_male', 'deck', 'embark_town', 'alive', 'class', 'alone'])
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True, dtype=float)

# 3. データの分割
train_df, predict_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['survived']
)

# 4. CSVファイルとして保存
# ▼▼▼▼▼▼▼▼▼▼ ここが今回の修正箇所 ▼▼▼▼▼▼▼▼▼▼
# インデックスをリセットして、0からの連番を新しいインデックス列('index')として付与します。
train_df.reset_index(drop=True).to_csv('titanic_train.csv', index_label='index')
predict_df.reset_index(drop=True).to_csv('titanic_predict.csv', index_label='index')
# ▲▲▲▲▲▲▲▲▲▲ ここまで修正 ▲▲▲▲▲▲▲▲▲▲

print("titanic_train.csv と titanic_predict.csv を作成しました。")