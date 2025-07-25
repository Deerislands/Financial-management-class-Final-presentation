from flask import Flask, request, jsonify
import pandas as pd
import io
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # ローカルHTMLからのアクセスを許可

# ✅ OpenAI APIクライアントの初期化
client = OpenAI(api_key="API key")

@app.route("/get_comment", methods=["POST"])
def get_comment():
    try:
        csv_text = request.json.get("csv", "")
        df = pd.read_csv(io.StringIO(csv_text))

        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df = df.dropna(subset=['cost'])
        df['genre'] = df['genre'].str.strip()
        df['day'] = pd.to_datetime(df['day'])

        total_spending = int(df['cost'].sum())
        avg_spending = int(df.groupby('day')['cost'].sum().mean())

        df['month'] = df['day'].dt.to_period('M')
        monthly_summary = df.groupby('month')['cost'].sum().sort_index()
        monthly_labels = monthly_summary.index.astype(str).tolist()
        monthly_values = monthly_summary.astype(int).tolist()

        trend = "不明"
        if len(monthly_values) >= 2:
            trend = "増加" if monthly_values[-1] > monthly_values[-2] else "減少" if monthly_values[-1] < monthly_values[-2] else "安定"

        genre_summary = df.groupby('genre')['cost'].sum().sort_values(ascending=False)
        genre_total = genre_summary.sum()
        genre_percent = (genre_summary / genre_total * 100).round(1).astype(str) + '%'
        genre_summary.index = genre_summary.index.astype(str)
        genre_percent.index = genre_percent.index.astype(str)

        top_genres = "\n".join([
            f"- {genre}: ¥{int(genre_summary[genre])}（全体の{genre_percent[genre]}）"
            for genre in genre_summary.head(5).index
        ])

        daily_total = df.groupby('day')['cost'].sum()
        max_day = daily_total.idxmax().strftime("%Y-%m-%d")
        max_value = int(daily_total.max())

        prompt = f"""
あなたは家庭の支出アドバイザーです。以下の支出分析に基づいて、自然で流暢な**日本語**でコメントとアドバイスを提供してください。

【支出概要】
- 総支出：¥{total_spending}
- 1日あたりの平均支出：¥{avg_spending}
- 月ごとの支出傾向：{" → ".join([f"{m} ¥{v}" for m, v in zip(monthly_labels, monthly_values)])}（傾向：{trend}）

【ジャンル別支出】
{top_genres}

【特に支出が多かった日】
- 支出が最も多かった日：{max_day}（¥{max_value}）
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたは家庭の支出アドバイザーです。日本語で回答してください。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )

        comment = response.choices[0].message.content
        return jsonify({"comment": comment})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
