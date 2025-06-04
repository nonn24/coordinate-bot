from fastapi import FastAPI, Request,BackgroundTasks, Header
from fastapi.responses import FileResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageSendMessage
from starlette.exceptions import HTTPException
import os
import uuid
import random
from dotenv import load_dotenv
import linebot.v3.messaging as bot
import time
import linebot.v3.messaging
from linebot.v3.messaging.models.broadcast_request import BroadcastRequest
from linebot.v3.messaging.rest import ApiException
from pprint import pprint

load_dotenv()

app = FastAPI()
LINE_BOT_API = LineBotApi(os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
handler = WebhookHandler(os.environ["LINE_CHANNEL_SECRET"])


@app.get("/")
def root():
    return {"title": "Echo Bot"}

configuration = bot.Configuration(
    access_token=os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
)

@app.post("/callback")
async def callback(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature=Header(None),
):
    body = await request.body()

    try:
        background_tasks.add_task(
            handler.handle, body.decode("utf-8"), x_line_signature
        )
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return "ok"


@handler.add(MessageEvent)
def handle_message(event):
    if event.type != "message" or event.message.type != "text":
        return
    
    message_text = event.message.text.lower()
    
    if "プロフィール" in message_text:
        message = TextMessage(text="齊藤京子さんについて紹介します！\n齊藤京子\n1997年9月5日生\n4/5 齊藤京子卒業コンサート in 横浜スタジアム にて日向坂46を卒業\n5/1~ 東宝芸能所属")#ここにプロフィールを流すようにするあとでかくor写真を添付
        LINE_BOT_API.reply_message(event.reply_token, message)