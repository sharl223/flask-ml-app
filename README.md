# 🤖 AI Playground - 機械学習を簡単に体験

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-Live-brightgreen.svg)](https://flask-ml-app-g0mo.onrender.com/)

> **初心者でも簡単に機械学習を体験できるWebアプリケーション**  
> **Easy-to-use machine learning web application for beginners**

## 🌟 特徴 / Features

### 🎯 **初心者フレンドリー / Beginner-Friendly**
- **日本語・英語対応** / Japanese & English support
- **直感的なUI** / Intuitive user interface
- **ステップバイステップガイド** / Step-by-step guidance
- **詳細な解説付き** / Detailed explanations

### 🚀 **多様な機械学習機能 / Diverse ML Features**
- **線形回帰** / Linear Regression
- **ロジスティック回帰** / Logistic Regression  
- **LightGBM** / LightGBM Machine Learning
- **CSV加工集計** / CSV Processing & Aggregation

### 💡 **高度な分析機能 / Advanced Analytics**
- **SHAP分析** / SHAP Analysis for model interpretation
- **特徴量重要度** / Feature importance visualization
- **予測結果の可視化** / Prediction result visualization
- **データ前処理** / Data preprocessing

## 🎮 デモ / Demo

**🌐 ライブデモ: [https://flask-ml-app-g0mo.onrender.com/](https://flask-ml-app-g0mo.onrender.com/)**

### 📱 スクリーンショット / Screenshots

<details>
<summary>🏠 ホームページ / Homepage</summary>

![Homepage](https://via.placeholder.com/800x400/667eea/ffffff?text=AI+Playground+Homepage)

- 直感的なナビゲーション
- 機能の概要説明
- クイックスタートガイド

</details>

<details>
<summary>📊 線形回帰 / Linear Regression</summary>

![Linear Regression](https://via.placeholder.com/800x400/764ba2/ffffff?text=Linear+Regression+Page)

- データアップロード
- モデル学習
- 予測実行
- 結果可視化

</details>

<details>
<summary>🎯 ロジスティック回帰 / Logistic Regression</summary>

![Logistic Regression](https://via.placeholder.com/800x400/667eea/ffffff?text=Logistic+Regression+Page)

- 分類問題の解決
- モデル性能評価
- SHAP分析

</details>

<details>
<summary>🌳 LightGBM</summary>

![LightGBM](https://via.placeholder.com/800x400/764ba2/ffffff?text=LightGBM+Page)

- 高性能機械学習
- 複雑なパターン検出
- 高度な予測精度

</details>

<details>
<summary>📋 CSV加工集計 / CSV Processing</summary>

![CSV Processing](https://via.placeholder.com/800x400/667eea/ffffff?text=CSV+Processing+Page)

- データ前処理
- 統計分析
- ファイル変換

</details>

## 🛠️ 技術スタック / Tech Stack

### **バックエンド / Backend**
- **Python 3.8+** - メイン言語
- **Flask 2.0+** - Webフレームワーク
- **Pandas** - データ処理
- **NumPy** - 数値計算
- **Scikit-learn** - 機械学習ライブラリ
- **LightGBM** - 勾配ブースティング
- **SHAP** - モデル解釈

### **フロントエンド / Frontend**
- **Bootstrap 5** - UIフレームワーク
- **Bootstrap Icons** - アイコン
- **JavaScript** - インタラクティブ機能
- **Chart.js** - グラフ描画

### **インフラ / Infrastructure**
- **Render** - ホスティング
- **Flask-Session** - セッション管理
- **Joblib** - モデル保存

## 🚀 クイックスタート / Quick Start

### **1. リポジトリのクローン / Clone Repository**
```bash
git clone https://github.com/your-username/ai-playground.git
cd ai-playground
```

### **2. 仮想環境の作成 / Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### **3. 依存関係のインストール / Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. アプリケーションの起動 / Start Application**
```bash
python app.py
```

### **5. ブラウザでアクセス / Access in Browser**
```
http://localhost:5000
```

## 📖 使用方法 / How to Use

### **🎯 線形回帰 / Linear Regression**
1. **データアップロード** / Upload your CSV data
2. **目的変数選択** / Select target variable
3. **特徴量選択** / Select features
4. **モデル学習** / Train model
5. **予測実行** / Make predictions
6. **結果分析** / Analyze results

### **🎯 ロジスティック回帰 / Logistic Regression**
1. **データアップロード** / Upload your CSV data
2. **分類対象選択** / Select classification target
3. **特徴量選択** / Select features
4. **モデル学習** / Train model
5. **分類予測** / Make classifications
6. **精度評価** / Evaluate accuracy

### **🌳 LightGBM**
1. **データアップロード** / Upload your CSV data
2. **目的変数選択** / Select target variable
3. **特徴量選択** / Select features
4. **高性能学習** / High-performance training
5. **予測実行** / Make predictions
6. **詳細分析** / Detailed analysis

### **📋 CSV加工集計 / CSV Processing**
1. **ファイルアップロード** / Upload CSV file
2. **インデックス設定** / Set index options
3. **処理内容選択** / Select processing options
4. **実行** / Execute processing
5. **結果ダウンロード** / Download results

## 📊 対応データ形式 / Supported Data Formats

### **入力形式 / Input Formats**
- **CSV** - カンマ区切り値
- **UTF-8** - 文字エンコーディング
- **最大10MB** - ファイルサイズ制限

### **出力形式 / Output Formats**
- **CSV** - 処理結果
- **PNG** - グラフ・チャート
- **JSON** - API応答

## 🌍 多言語対応 / Multi-language Support

### **日本語 / Japanese**
- 完全な日本語インターフェース
- 日本語エラーメッセージ
- 日本語ドキュメント

### **English**
- Complete English interface
- English error messages
- English documentation

## 🔧 カスタマイズ / Customization

### **テーマの変更 / Theme Customization**
```css
/* static/style.css */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #f093fb;
}
```

### **機能の追加 / Feature Extension**
```python
# apps/custom_model.py
from flask import Blueprint

custom_bp = Blueprint('custom', __name__)

@custom_bp.route('/custom_model')
def custom_model():
    # カスタムモデルの実装
    pass
```

## 📈 パフォーマンス / Performance

### **最適化機能 / Optimization Features**
- **レスポンシブデザイン** / Responsive design
- **遅延読み込み** / Lazy loading
- **キャッシュ機能** / Caching
- **圧縮転送** / Gzip compression

### **セキュリティ / Security**
- **HTTPS通信** / HTTPS communication
- **入力検証** / Input validation
- **セッション管理** / Session management
- **ファイル制限** / File restrictions

## 🤝 コントリビューション / Contributing

### **開発環境のセットアップ / Development Setup**
```bash
# 開発用依存関係のインストール
pip install -r requirements-dev.txt

# テストの実行
python -m pytest tests/

# コードフォーマット
black .
flake8 .
```

### **プルリクエスト / Pull Requests**
1. **フォーク** / Fork the repository
2. **ブランチ作成** / Create a feature branch
3. **変更実装** / Implement changes
4. **テスト実行** / Run tests
5. **プルリクエスト作成** / Create pull request

## 📝 ライセンス / License

このプロジェクトは **MIT License** の下で公開されています。

```
MIT License

Copyright (c) 2025 AI Playground

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📞 サポート / Support

### **お問い合わせ / Contact**
- **Email**: support@ai-playground.com
- **GitHub Issues**: [Issue Tracker](https://github.com/your-username/ai-playground/issues)
- **Discord**: [AI Playground Community](https://discord.gg/ai-playground)

### **ドキュメント / Documentation**
- **公式サイト**: [https://flask-ml-app-g0mo.onrender.com/](https://flask-ml-app-g0mo.onrender.com/)
- **API ドキュメント**: `/api/docs`
- **チュートリアル**: `/tutorial`

## 🙏 謝辞 / Acknowledgments

### **オープンソースライブラリ / Open Source Libraries**
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Scikit-learn](https://scikit-learn.org/) - Machine learning
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting
- [SHAP](https://shap.readthedocs.io/) - Model interpretation
- [Bootstrap](https://getbootstrap.com/) - UI framework

### **コミュニティ / Community**
- **データサイエンスコミュニティ** / Data Science Community
- **機械学習エンジニア** / Machine Learning Engineers
- **オープンソースコントリビューター** / Open Source Contributors

## 📊 統計 / Statistics

![GitHub stars](https://img.shields.io/github/stars/your-username/ai-playground)
![GitHub forks](https://img.shields.io/github/forks/your-username/ai-playground)
![GitHub issues](https://img.shields.io/github/issues/your-username/ai-playground)
![GitHub pull requests](https://img.shields.io/github/issues-pr/your-username/ai-playground)

## 🌟 スターを付ける / Star this Repository

このプロジェクトが役に立った場合は、⭐ **スター** を付けてください！

If this project helped you, please give it a ⭐ **star**!

---

**Made with ❤️ by AI Playground Team**

*初心者でも簡単に機械学習を体験できる世界を目指して*  
*Aiming for a world where anyone can easily experience machine learning* 