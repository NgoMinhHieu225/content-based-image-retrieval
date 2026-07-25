import os
import time
import streamlit as st
from PIL import Image
from feature_extractor import FeatureExtractor
from retriever import ImageRetriever

st.set_page_config(page_title="AI Visual Product Search", page_icon="🛍️", layout="wide")

st.title("🛍️ AI Visual Product Recommendation Engine")
st.markdown("---")
st.write("Hệ thống tìm kiếm sản phẩm tương đồng sử dụng **ResNet50 Deep Learning** & **Meta FAISS Vector Database**.")


@st.cache_resource
def load_engines():
    return FeatureExtractor(), ImageRetriever()


try:
    extractor, retriever = load_engines()

    st.sidebar.header("📁 Tải ảnh sản phẩm cần tìm")
    uploaded_file = st.sidebar.file_uploader("Chọn 1 bức ảnh (giày, áo, laptop...)",
                                             type=["jpg", "png", "jpeg", "webp", "bmp", "jfif"])

    if uploaded_file is not None:
        col_query, col_results = st.columns([1, 3])

        with col_query:
            st.subheader("🔍 Ảnh truy vấn")
            query_image = Image.open(uploaded_file)
            st.image(query_image, width="stretch")

            temp_path = "temp_query_image.png"
            query_image.save(temp_path)

        with col_results:
            st.subheader("💡 Top Sản Phẩm Tương Đồng Nhất")

            with st.spinner("⏳ AI đang trích xuất vector và truy vấn FAISS..."):
                start_time = time.time()
                query_vector = extractor.extract(temp_path)
                results = retriever.search(query_vector, top_k=5)
                end_time = time.time()
                search_latency = (end_time - start_time) * 1000  # Đổi sang mili-giây

            # Hiển thị thông số hiệu năng (Điểm cộng lớn khi phỏng vấn)
            st.info(
                f"⚡ **Tốc độ truy vấn FAISS:** `{search_latency:.2f} ms` | 🗂️ **Quét qua kho dữ liệu:** `{retriever.index.ntotal} sản phẩm`")

            res_cols = st.columns(len(results))
            for i, res in enumerate(results):
                with res_cols[i]:
                    res_img = Image.open(res["image_path"])
                    st.image(res_img, width="stretch")
                    st.success(f"**Giống: {res['score']:.1f}%**")
                    st.caption(f"📁 {os.path.basename(res['image_path'])}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

except Exception as e:
    st.error(
        f"⚠️ Có lỗi xảy ra hoặc chưa tìm thấy file index: {e}. Vui lòng chạy lại lệnh `python feature_extractor.py`!")