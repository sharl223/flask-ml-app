# ベースとなる公式Pythonイメージを指定します。'slim'は軽量版です。
FROM python:3.10-slim

# LightGBMが必要とするOSのライブラリ(libgomp1)をインストール
RUN apt-get update && apt-get install -y libgomp1

# コンテナ内での作業ディレクトリを設定します。
WORKDIR /app

# まずrequirements.txtだけをコピーし、ライブラリをインストールします。
# こうすることで、アプリのコードだけを変更した場合に、ライブラリの再インストールが不要になり、ビルドが高速化します。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションの全てのコードをコンテナ内にコピーします。
COPY . .

# コンテナが起動したときに実行するコマンドを指定します。
# ここで、本番用サーバーGunicornを使ってアプリを起動します。
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "app:app"]