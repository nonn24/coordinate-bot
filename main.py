from fastapi import FastAPI, Request,BackgroundTasks, Header
from fastapi.responses import FileResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageSendMessage, ImageMessage
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
import cv2
import numpy as np
from skimage import feature, color
import tempfile
from pathlib import Path

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
    if event.type != "message":
        return
    
    # テキストメッセージの処理
    if event.message.type == "text":
        message_text = event.message.text.lower()
        
        if "プロフィール" in message_text:
            message = TextMessage(text="齊藤京子さんについて紹介します！\n齊藤京子\n1997年9月5日生\n4/5 齊藤京子卒業コンサート in 横浜スタジアム にて日向坂46を卒業\n5/1~ 東宝芸能所属")#ここにプロフィールを流すようにするあとでかくor写真を添付
            LINE_BOT_API.reply_message(event.reply_token, message)
    
    # 画像メッセージの処理
    elif event.message.type == "image":
        try:
            # 画像をダウンロード
            image_path = download_image_from_line(event.message.id)
            
            # 画像から特徴量を抽出
            features = extract_clothing_features(image_path)
            
            # 特徴量から簡単な説明を生成
            dominant_colors = features["dominant_colors"]
            color_description = "主な色: " + ", ".join([f"RGB{color['rgb']}（{color['percentage']:.1f}%）" for color in dominant_colors])
            
            shape_features = features["shape_features"]
            texture_desc = "テクスチャ特徴: " + ("複雑" if len(features["texture_features"]["hog_features"]) > 10 else "シンプル")
            
            aspect_desc = ""
            if shape_features["aspect_ratio"] > 1.5:
                aspect_desc = "横長の"
            elif shape_features["aspect_ratio"] < 0.67:
                aspect_desc = "縦長の"
            
            # 返信メッセージを作成
            reply_text = f"服の解析結果:\n{color_description}\n{texture_desc}\n{aspect_desc}アイテムです。"
            
            # 画像の処理が終わったら一時ファイルを削除
            os.remove(image_path)
            
            # 結果を返信
            LINE_BOT_API.reply_message(event.reply_token, TextMessage(text=reply_text))
            
        except Exception as e:
            import traceback
            print(f"Error processing image: {e}")
            print(traceback.format_exc())
            LINE_BOT_API.reply_message(
                event.reply_token,
                TextMessage(text=f"画像の処理中にエラーが発生しました: {str(e)}")
            )


# 画像処理のための定数
TEMP_IMAGE_DIR = Path("./temp_images")

def download_image_from_line(message_id):
    """LINEから画像をダウンロードして一時ファイルとして保存する"""
    message_content = LINE_BOT_API.get_message_content(message_id)
    
    # 一時ファイル名を生成（UUIDを使用して一意のファイル名を生成）
    temp_file_path = TEMP_IMAGE_DIR / f"{uuid.uuid4()}.jpg"
    
    with open(temp_file_path, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
            
    return str(temp_file_path)

def extract_clothing_features(image_path):
    """画像から衣類の特徴量を抽出する関数"""
    # 画像の読み込み
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "画像を読み込めませんでした"}
    
    # サイズを正規化
    img = cv2.resize(img, (300, 300))
    
    # 色特徴（色ヒストグラム）
    color_features = extract_color_features(img)
    
    # テクスチャ特徴（HOG特徴量）
    texture_features = extract_texture_features(img)
    
    # 形状特徴（エッジ情報）
    shape_features = extract_shape_features(img)
    
    # 結果をまとめる
    result = {
        "color_features": color_features,
        "texture_features": texture_features,
        "shape_features": shape_features,
        "dominant_colors": get_dominant_colors(img, 3)
    }
    
    return result

def extract_color_features(img):
    """画像から色の特徴量を抽出する"""
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 色ヒストグラムを計算
    h_bins = 8
    s_bins = 8
    v_bins = 8
    
    # ヒストグラムのビン
    h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
    
    # 正規化
    h_hist = cv2.normalize(h_hist, h_hist).flatten().tolist()
    s_hist = cv2.normalize(s_hist, s_hist).flatten().tolist()
    v_hist = cv2.normalize(v_hist, v_hist).flatten().tolist()
    
    return {
        "hue_histogram": h_hist,
        "saturation_histogram": s_hist,
        "value_histogram": v_hist
    }

def extract_texture_features(img):
    """画像からテクスチャの特徴量を抽出する"""
    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG特徴量を計算
    hog_features = feature.hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        block_norm='L2-Hys'
    ).tolist()
    
    # LBP (Local Binary Pattern) も計算
    radius = 3
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # 正規化
    
    return {
        "hog_features": hog_features[:20],  # 長くなりすぎるので最初の20個だけ
        "lbp_histogram": lbp_hist.tolist()
    }

def extract_shape_features(img):
    """画像から形状の特徴量を抽出する"""
    # エッジ検出
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 輪郭の特徴量
    if len(contours) > 0:
        # 最大の輪郭を見つける
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        perimeter = cv2.arcLength(max_contour, True)
        
        # 境界ボックス
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "aspect_ratio": float(aspect_ratio),
            "contour_count": len(contours)
        }
    else:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "aspect_ratio": 0.0,
            "contour_count": 0
        }

def get_dominant_colors(img, k=3):
    """画像から主要な色を抽出する"""
    # 画像をRGBに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 画像をピクセルのリストに変換
    pixels = img_rgb.reshape(-1, 3)
    
    # k-meansクラスタリングを実行
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 各クラスタのピクセル数をカウント
    counts = np.bincount(labels.flatten())
    
    # RGB値と割合のリストを作成
    dominant_colors = []
    for i in range(k):
        rgb = centers[i].astype(int).tolist()
        percentage = float(counts[i]) / len(labels) * 100
        dominant_colors.append({
            "rgb": rgb,
            "percentage": percentage
        })
    
    # 割合の降順でソート
    dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)
    
    return dominant_colors