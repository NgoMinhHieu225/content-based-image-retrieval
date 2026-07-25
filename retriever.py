import os
import pickle
import numpy as np
import faiss


class ImageRetriever:
    def __init__(self, feature_path="features/features.pkl"):
        self.feature_path = feature_path
        self.image_paths = []
        self.index = None
        self.load_features()

    def load_features(self):
        """Tải vector từ file và xây dựng chỉ mục tìm kiếm FAISS Index tốc độ cao"""
        if not os.path.exists(self.feature_path):
            raise FileNotFoundError(
                f"⚠️ Không tìm thấy file {self.feature_path}. Vui lòng chạy feature_extractor.py trước!")

        with open(self.feature_path, "rb") as f:
            data = pickle.load(f)

        self.image_paths = data["paths"]
        features = np.array(data["features"]).astype("float32")

        # Chuẩn hóa L2 (L2 Normalization) cho vector để Inner Product tương đương với Cosine Similarity
        faiss.normalize_L2(features)

        # Khởi tạo FAISS Index với thuật toán Inner Product (IP) cho độ dài vector 2048 của ResNet50
        dimension = features.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Nạp toàn bộ vector vào kho chỉ mục của FAISS
        self.index.add(features)
        print(f"🚀 [FAISS ENGINE] Đã nạp thành công {self.index.ntotal} vector vào cơ sở dữ liệu tốc độ cao!")

    def search(self, query_vector, top_k=5):
        """Truy vấn siêu tốc Top-K sản phẩm tương đồng bằng FAISS"""
        if self.index is None:
            raise ValueError("⚠️ Index chưa được khởi tạo!")

        # Chuẩn hóa vector truy vấn theo chuẩn float32 và L2 Norm
        query_vector = np.array([query_vector]).astype("float32")
        faiss.normalize_L2(query_vector)

        # Tìm kiếm trong FAISS: trả về điểm số và vị trí
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx == -1:
                continue

            # Đổi từ điểm số sang tỷ lệ % (0 đến 100%)
            raw_score = scores[0][i]
            score_pct = max(0.0, float(raw_score) * 100)

            results.append({
                "image_path": self.image_paths[idx],
                "score": score_pct
            })

        return results