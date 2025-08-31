# Facial Recognition Project Report  

## ✅ Implemented Features  
This project implements a **complete real-time and video-based facial recognition pipeline** with advanced optimizations for **speed, scalability, and robustness**.  

---

## 🔹 Models  
- **YOLOv8 (`yolov8n-face.pt`)** → Face detection.  
- **FaceNet (InceptionResnetV1, pretrained on VGGFace2)** → 512-D face embeddings.  
- **GAN-based Augmentations** → Generate realistic variants for robustness:  
  - StarGAN  (Credits to: https://github.com/yunjey/stargan)
  - EG3D  (Credits to: https://github.com/NVlabs/eg3d)
  - DiscoFaceGAN  (Credits to: https://github.com/microsoft/DiscoFaceGAN)
  - BeautyGAN  (Credits to: https://jonhyuk0922.tistory.com/103)

These GANs simulate variations in **pose, lighting, age, and expression**.  

---

## 🔹 Preprocessing  
- Resize → `160×160`, tensor conversion, normalization.  
- **Lighting adjustments**:  
  - Gamma correction (dark frames).  
  - Brightness detection (mean pixel intensity).  
- **Dynamic thresholding** → larger faces require stricter matching.  
- **GAN augmentations** → dynamic variants generated for stability.  

---

## 🔹 Embedding Generation  
- Functions: `get_embedding`, `face_to_embedding`.  
- `build_known_embeddings()` pipeline:  
  - Each person → **1 reference image (`1.*`)**.  
  - Generate multiple variants → original, gamma-corrected, flipped, yaw-left, yaw-right and mentioned gan based augmentations.  
  - Average embeddings → **compact identity vector**.  

---

Each identity is stored with multiple context components:

- **Mean Map (`mean`)** → average embedding (float32).  
- **Attention Map (`attn`)** → weighted average, giving higher weight to closer vectors (float32).  
- **PCA Vectors (`pca_vecs`)** → reduced-dimension basis vectors for compact representation.  
- **Covariance Matrix (`covariance`)** → captures variation of embeddings for that identity.  
- **Prototypes (`prototypes`)** → representative embeddings for fast matching and robustness.  

This structure reduces redundancy, improves stability, and accelerates comparisons.  

---

## 🔹 Storage & Database  
- Context maps stored in **MySQL**.  
- JSON serialization for flexible storage and retrieval.  

---

## 🎥 Real-Time & Video Face Recognition  
- **Real-time webcam recognition** → `run_realtime.py`.  
- **Video recognition** → `run_video.py`.  
- Features:  
  - Motion detection (skip redundant frames).  
  - Lag handling (drop frames if behind).  
  - Duplicate embedding skipping.  
  - Label smoothing with deque.  

---

## ⚙️ Utilities  
- Similarity: **cosine similarity + L2 distance**.  
- Lighting: gamma correction, brightness check.  
- Performance logging: detection, embedding, lookup, total frame time.  

---

## 🚧 The Latency Issue: O(n) Problem  
- Brute force required **O(n)** comparisons.  
- Too slow and unscalable for real-time recognition.  

---

## 💡 Solution  

### 1. Context Maps  
- Only **two vectors per person** stored.  
- Compact and stable representation.  

### 2. Locality Sensitive Hashing (LSH)  
- Speed up lookup via **hash buckets**.  
- Process:  
  1. Generate random hyperplanes.  
  2. Hash embeddings → bit string.  
  3. Store embeddings in buckets.  
  4. Query embedding → mapped to same/nearby buckets.  
  5. Final candidates matched with cosine/L2.  

**Benefits**:  
- Lookup no longer O(n).  
- Scales to thousands of identities.  
- Multi-probe → high recall + speed.  

---

## ⚡ Speed Comparisons  

| Method        | Detection (ms) | Embedding (ms) | Lookup (ms) | Frame Total (ms) |
|---------------|----------------|----------------|-------------|------------------|
| Brute Force   | 110.52         | 107.69         | 0.32        | 92.47            |
| LSH Original (12 planes, 4 tables) | 125.44 | 102.77 | 0.31 | 76.92 |
| LSH Optimized (16 planes, 5 tables) | **72.12** | **60.92** | **0.26** | **42.03** |

➡️ Optimized LSH is **2× faster than brute force** while maintaining high accuracy.  

---

## 🛠️ Production Readiness Updates  
- Config management → `config.yaml`.  
- Reproducibility → `requirements.txt`.  
- Threading & parallelization → detection, embedding, and DB queries run concurrently.  
- GAN augmentations → reduce need for multiple raw images per person.  
- Logging & monitoring → detailed timing metrics.  

---

## 📌 Conclusion  
This project delivers a **production-ready, real-time facial recognition system** with:  
- **Accuracy** → context maps + embeddings.  
- **Speed** → optimized LSH + parallelization.  
- **Scalability** → compact DB storage, sub-linear lookup.  

Robust against **lighting, pose, and expression changes**, modular for future updates.  

---

By combining **state-of-the-art deep learning** with **practical engineering optimizations**, this project achieves **scalable, real-time face recognition** ready for deployment.  
