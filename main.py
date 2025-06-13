from fastapi import FastAPI, Request,BackgroundTasks, Header
from fastapi.responses import FileResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageSendMessage, ImageMessage
from starlette.exceptions import HTTPException
import os
from dotenv import load_dotenv
import linebot.v3.messaging as bot
from pathlib import Path
import uuid
import cv2
import numpy as np
from skimage import feature

# ユースケース関連のインポート
from usecase.clothing_analysis import (
    download_image_from_line,
    extract_clothing_features,
    generate_clothing_description
)
from color_utils import get_color_name

# 環境変数の読み込み
load_dotenv(override=True)

# アクセストークンとシークレットの確認
channel_token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
channel_secret = os.environ.get("LINE_CHANNEL_SECRET")

if not channel_token or not channel_secret:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN または LINE_CHANNEL_SECRET が設定されていません。")

app = FastAPI()
LINE_BOT_API = LineBotApi(channel_token)
handler = WebhookHandler(channel_secret)


@app.get("/")
def root():
    return {"title": "Echo Bot"}

# 変数から直接設定を使う（環境変数から直接取得するよりも安全）
configuration = bot.Configuration(
    access_token=channel_token
)

@app.post("/callback")
async def callback(request: Request, background_tasks: BackgroundTasks):
    # リクエストボディを取得
    body = await request.body()
    body_text = body.decode("utf-8")
    
    # X-Line-Signatureヘッダーを取得
    signature = request.headers.get("X-Line-Signature", "")
    
    try:
        # バックグラウンドタスクとして実行して応答速度を向上
        background_tasks.add_task(handler.handle, body_text, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return "ok"


@handler.add(MessageEvent)
def handle_message(event):
    if event.type != "message":
        return
    
    # テキストメッセージの処理
    if event.message.type == "text":
        message_text = event.message.text.lower()
        
        if "やり方" in message_text:
            message = TextMessage(text="写真を添付すると色解析を行います！")#ここにプロフィールを流すようにするあとでかくor写真を添付
            LINE_BOT_API.reply_message(event.reply_token, message)
    
    # 画像メッセージの処理
    elif event.message.type == "image":
        try:
            # 画像をダウンロード            
            image_path = download_image_from_line(LINE_BOT_API, event.message.id, TEMP_IMAGE_DIR)
            
            # 画像から特徴量を抽出
            features = extract_clothing_features(image_path)
            
            # 特徴量から簡単な説明を生成
            reply_text = generate_clothing_description(features)
            
            # 画像の処理が終わったら一時ファイルを削除
            os.remove(image_path)
            
            # 結果を返信
            LINE_BOT_API.reply_message(event.reply_token, TextMessage(text=reply_text))
            
        except Exception as e:
            LINE_BOT_API.reply_message(
                event.reply_token,
                TextMessage(text=f"画像の処理中にエラーが発生しました。: {str(e)}")
            )


TEMP_IMAGE_DIR = Path("./temp_images")