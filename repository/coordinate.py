from sqlalchemy import create_engine, text
from datetime import datetime
import json

class CoordinateRepository:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    def get_coordinate(self, coordinate_id):
        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT * FROM coordinate WHERE id = :id"),
                {"id": coordinate_id}
            )
            return result.fetchone()

    def add_coordinate(self, date=None):
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        with self.engine.connect() as connection:
            result = connection.execute(
                text("INSERT INTO coordinate (date) VALUES (:date) RETURNING id"),
                {"date": date}
            )
            return result.fetchone()[0] 
            
    def update_coordinate(self, coordinate_id, date):
        with self.engine.connect() as connection:
            connection.execute(
                text("UPDATE coordinate SET date = :date WHERE id = :id"),
                {"date": date, "id": coordinate_id}
            )
            
    def delete_coordinate(self, coordinate_id):
        with self.engine.connect() as connection:
            connection.execute(
                text("DELETE FROM coordinate WHERE id = :id"),
                {"id": coordinate_id}
            )
    
    def save_clothing_features(self, coordinate_id, features):
        """服の特徴量をデータベースに保存する"""
        # 特徴量をJSON形式に変換
        features_json = json.dumps(features)
        
        with self.engine.connect() as connection:
            # clothingテーブルがあることを前提としています
            # 必要に応じてテーブル構造に合わせて修正してください
            connection.execute(
                text("INSERT INTO clothing (coordinate_id, features) VALUES (:coordinate_id, :features)"),
                {"coordinate_id": coordinate_id, "features": features_json}
            )
    
    def get_clothing_by_coordinate(self, coordinate_id):
        """コーディネートIDに関連する服の情報を取得する"""
        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT * FROM clothing WHERE coordinate_id = :coordinate_id"),
                {"coordinate_id": coordinate_id}
            )
            return result.fetchall()
    
    def find_similar_clothing(self, features, limit=5):
        """特徴量が類似している服を検索する
        注意: この実装は簡易的なものです。実際には類似度計算のロジックが必要です。
        """
        # 全ての服の特徴量を取得
        with self.engine.connect() as connection:
            result = connection.execute(text("SELECT id, features FROM clothing"))
            all_clothing = result.fetchall()
            
        # ここでは単純化のために、実装の詳細は省略しています
        # 実際には、色ヒストグラムの比較や特徴量ベクトル間の距離計算などが必要です
        # これは別のユーティリティモジュールとして実装することをお勧めします
            
        return []  # 類似度が高い順に服のIDを返す