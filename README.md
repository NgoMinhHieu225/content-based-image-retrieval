# 🛍️ AI Visual Product Recommendation Engine

An end-to-end Content-Based Image Retrieval (CBIR) system engineered to power visual product search for e-commerce platforms. The system leverages Deep Learning (**ResNet50**) for high-dimensional feature extraction and mathematical vector similarity algorithms (**Cosine Similarity**) for real-time item recommendations.

## 🌟 Demo Preview

![Uploading demo_web.png…]()


## 🚀 Key Features & Architecture
* **Deep Learning Feature Extraction:** Utilizes a PyTorch pretrained **ResNet50** model (removing the classification layer) to convert visual product data into robust 2048-dimensional mathematical feature vectors.
* **Vector Similarity Search Engine:** Implements customized **Cosine Similarity** algorithms via **NumPy** to index and retrieve visually analogous items with high precision and low computational latency.
* **Modern Web Interface:** Built an interactive, dark-mode web application using **Streamlit**, allowing users to upload query images and instantly discover top-5 similar products in real time.
* **RAG & Vector DB Foundation:** Demonstrates practical understanding of high-dimensional space indexing and retrieval pipelines — the core mechanics behind modern Vector Databases (FAISS, Qdrant) and RAG architectures.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Deep Learning Framework:** PyTorch, Torchvision (ResNet50)
* **Mathematical Computation:** NumPy
* **Image Processing:** PIL (Pillow)
* **Frontend / GUI:** Streamlit

## ⚙️ Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NgoMinhHieu225/content-based-image-retrieval.git](https://github.com/NgoMinhHieu225/content-based-image-retrieval.git)
   cd content-based-image-retrieval
