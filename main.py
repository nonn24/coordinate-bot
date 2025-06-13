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
            # 色名に変換
            color_names = []
            for color in dominant_colors:
                rgb = color['rgb']
                color_name = get_color_name(rgb)
                color_names.append(f"{color_name}（{color['percentage']:.1f}%）")
                
            color_description = "主な色: " + ", ".join(color_names)
            
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
            LINE_BOT_API.reply_message(
                event.reply_token,
                TextMessage(text="画像の処理中にエラーが発生しました。")
            )


# 画像処理のための定数
TEMP_IMAGE_DIR = Path("./temp_images")
# ディレクトリが存在しない場合は作成
TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

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
    
    # 服の領域を検出
    clothing_region, mask = detect_clothing_region(img)
    
    # 服の領域が検出できなかった場合は元の画像を使用
    if clothing_region is None:
        print("服の領域が検出できなかったため、画像全体を解析します")
        clothing_region = img
        mask = None
    
    # マスク領域に黒でない部分があるか確認（有効なマスクかどうか）
    if mask is not None and np.sum(mask) > 0:
        # 色特徴（色ヒストグラム）- マスク領域のみ
        color_features = extract_color_features(clothing_region)
        
        # テクスチャ特徴（HOG特徴量）- マスク領域のみ
        texture_features = extract_texture_features(clothing_region)
        
        # 主要な色を抽出 - マスク領域のみ
        dominant_colors = get_dominant_colors(clothing_region, 3)
    else:
        # マスクが無効な場合は元の画像全体を使用
        color_features = extract_color_features(img)
        texture_features = extract_texture_features(img)
        dominant_colors = get_dominant_colors(img, 3)
    
    # 形状特徴（エッジ情報）
    shape_features = extract_shape_features(img, mask)
    
    # 結果をまとめる
    result = {
        "color_features": color_features,
        "texture_features": texture_features,
        "shape_features": shape_features,
        "dominant_colors": dominant_colors
    }
    
    return result

def extract_color_features(img):
    """画像から色の特徴量を抽出する"""
    # 黒（[0,0,0]）のピクセルをマスクとして扱う
    # OpenCVはBGR形式で画像を扱う
    mask = np.any(img != [0, 0, 0], axis=-1).astype(np.uint8) * 255
    
    # マスクに有効なピクセルがない場合（全て黒の場合）
    if np.sum(mask) == 0:
        mask = None  # マスクを無効化
    
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 色ヒストグラムを計算
    h_bins = 8
    s_bins = 8
    v_bins = 8
    
    # ヒストグラムのビン（マスクを使用）
    h_hist = cv2.calcHist([hsv], [0], mask, [h_bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [s_bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], mask, [v_bins], [0, 256])
    
    # 正規化
    if np.sum(h_hist) > 0:  # ヒストグラムが空でないことを確認
        h_hist = cv2.normalize(h_hist, h_hist).flatten().tolist()
        s_hist = cv2.normalize(s_hist, s_hist).flatten().tolist()
        v_hist = cv2.normalize(v_hist, v_hist).flatten().tolist()
    else:
        h_hist = [0] * h_bins
        s_hist = [0] * s_bins
        v_hist = [0] * v_bins
    
    return {
        "hue_histogram": h_hist,
        "saturation_histogram": s_hist,
        "value_histogram": v_hist
    }

def extract_texture_features(img):
    """画像からテクスチャの特徴量を抽出する"""
    # 黒（[0,0,0]）のピクセルをマスクとして扱う
    mask = np.any(img != [0, 0, 0], axis=-1)
    
    # マスクに有効なピクセルがあるか確認
    has_valid_pixels = np.any(mask)
    
    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # マスク領域のみ処理（マスクがある場合）
    if has_valid_pixels:
        # マスクを適用（背景を白に設定）
        masked_gray = gray.copy()
        masked_gray[~mask] = 255
        
        # HOG特徴量を計算（マスクされた領域のみ）
        hog_features = feature.hog(
            masked_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            block_norm='L2-Hys'
        ).tolist()
        
        # LBP (Local Binary Pattern) も計算
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(masked_gray, n_points, radius, method='uniform')
        
        # マスク領域のLBPのみを使用
        lbp_masked = lbp.copy()
        lbp_masked[~mask] = 0
        
        # ヒストグラムを計算
        lbp_hist, _ = np.histogram(lbp_masked[mask], bins=n_points + 2, range=(0, n_points + 2))
    else:
        # 有効なピクセルがない場合はデフォルト値
        hog_features = [0.0] * 20
        n_points = 8 * 3  # radius = 3
        lbp_hist = np.zeros(n_points + 2)
    
    # 正規化
    if lbp_hist.sum() > 0:
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    return {
        "hog_features": hog_features[:20],  # 長くなりすぎるので最初の20個だけ
        "lbp_histogram": lbp_hist.tolist()
    }

def detect_clothing_region(img):
    """画像から服と思われる最大の領域を検出する関数"""
    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 画像をぼかしてノイズを減らす
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # エッジ検出
    edges = cv2.Canny(blurred, 50, 150)
    
    # モルフォロジー演算でエッジを繋げる
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 輪郭検出
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # 最大の輪郭を見つける（服と想定）
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    
    # サイズが小さすぎる場合は無視（画像全体の5%以下）
    if area < (img.shape[0] * img.shape[1] * 0.05):
        return None, None
    
    # マスクを作成
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [max_contour], -1, 255, -1)
    
    # マスクを適用して服の部分だけを抽出
    clothing_region = cv2.bitwise_and(img, img, mask=mask)
    
    return clothing_region, mask

def extract_shape_features(img, mask=None):
    """画像から形状の特徴量を抽出する"""
    # エッジ検出
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # マスクが提供されている場合はそれを使用
    if mask is not None:
        edges = cv2.Canny(mask, 100, 200)
    else:
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
    # 黒（[0,0,0]）のピクセルを除外
    mask = np.any(img != [0, 0, 0], axis=-1)
    
    # 黒以外のピクセルが十分にあるか確認
    if np.sum(mask) < 100:  # 少なくとも100ピクセルが必要
        # マスクが不十分なら、元の画像の全ピクセルを使用
        mask = np.ones(img.shape[:2], dtype=bool)
    
    # 画像をRGBに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # マスクされたピクセルのみを取得
    pixels = img_rgb[mask]
    
    # ピクセルがない場合のエラー処理
    if len(pixels) == 0:
        return [{"rgb": [0, 0, 0], "percentage": 100.0}]
    
    # k-meansクラスタリングを実行
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # kがピクセル数より大きい場合は調整
    k = min(k, len(pixels))
    
    # k-means実行（少なくとも1つのクラスタは必要）
    if k >= 1:
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 各クラスタのピクセル数をカウント
        counts = np.bincount(labels.flatten())
        
        # RGB値と割合のリストを作成
        dominant_colors = []
        for i in range(k):
            rgb = centers[i].astype(int).tolist()
            percentage = float(counts[i]) / len(labels) * 100
            
            # [0,0,0]に近い黒は無視（背景の可能性）
            if sum(rgb) > 30:  # 合計値が30以上なら有効な色と判断
                dominant_colors.append({
                    "rgb": rgb,
                    "percentage": percentage
                })
        
        # 割合の降順でソート
        dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)
        
        # 有効な色が得られなかった場合
        if not dominant_colors:
            dominant_colors = [{"rgb": [128, 128, 128], "percentage": 100.0}]
        
        return dominant_colors
    else:
        # データが少なすぎる場合
        return [{"rgb": [128, 128, 128], "percentage": 100.0}]