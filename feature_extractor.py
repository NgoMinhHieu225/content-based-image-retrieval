import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class FeatureExtractor:
    def __init__(self):
        # Tự động ưu tiên dùng GPU nếu có, không thì dùng CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tải mô hình ResNet50 chuẩn với trọng lượng mới nhất
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)

        # Bỏ lớp Phân loại (Fully Connected layer) cuối cùng để lấy vector đặc trưng 2048 chiều
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # Tiền xử lý và chuẩn hóa ảnh theo đúng chuẩn của ResNet50
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image_path):
        """Trích xuất vector đặc trưng từ 1 bức ảnh"""
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.model(img_tensor).squeeze().cpu().numpy()

            # Chuẩn hóa L2 ngay tại khâu trích xuất cho FAISS
            return feature / np.linalg.norm(feature)
        except Exception as e:
            print(f"⚠️ Bỏ qua ảnh lỗi {image_path}: {e}")
            return None


def run_extraction():
    print("⏳ Đang khởi tạo AI Feature Extractor (ResNet50)...")
    extractor = FeatureExtractor()

    # Danh sách định dạng ảnh hỗ trợ (bao gồm file .jfif của bạn)
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif")
    image_paths = []

    # Quét toàn bộ dự án để tìm ảnh sản phẩm
    for root, dirs, files in os.walk("."):
        # Bỏ qua các thư mục hệ thống, môi trường ảo hoặc thư mục chứa vector
        if any(exclude in root for exclude in [".venv", "venv", ".git", "__pycache__", "features"]):
            continue
        for file in files:
            if file.lower().endswith(valid_extensions):
                # Bỏ qua ảnh demo giao diện hoặc ảnh query tạm thời
                if file in ["demoweb1.png", "temp_query_image.png"]:
                    continue
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("❌ LỖI: Không tìm thấy bức ảnh sản phẩm nào (.jpg, .png, .jfif...) trong dự án!")
        return

    print(f"🔍 Đã tìm thấy {len(image_paths)} bức ảnh sản phẩm. Đang tiến hành trích xuất vector...")

    features = []
    valid_paths = []
    for idx, path in enumerate(image_paths):
        print(f" ⚡ [{idx + 1}/{len(image_paths)}] Đang xử lý: {path}")
        feat = extractor.extract(path)
        if feat is not None:
            features.append(feat)
            valid_paths.append(path)

    # Tự động tạo thư mục features nếu chưa có
    os.makedirs("features", exist_ok=True)
    save_path = "features/features.pkl"

    with open(save_path, "wb") as f:
        pickle.dump({"paths": valid_paths, "features": features}, f)

    print(f"\n🎉 THÀNH CÔNG RỰC RỠ! Đã lưu vector của {len(valid_paths)} ảnh vào file: {save_path}")


if __name__ == "__main__":
    run_extraction()