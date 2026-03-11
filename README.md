# 🍣 Food Attractiveness Rater

โปรเจกต์วิเคราะห์และจัดอันดับ **ความน่ากิน** ของรูปอาหาร โดยใช้ Deep Learning ที่เรียนรู้จากรสนิยมการโหวตของมนุษย์จริงๆ

---

## 📌 Overview

โมเดลจำแนกรูปอาหารออกเป็น 2 คลาส:
- ✅ `attractive` — น่ากิน
- ❌ `unattractive` — ไม่น่ากิน

รองรับฟีเจอร์:
- ทำนายความน่ากินของรูปเดี่ยว (`predict_attractiveness`)
- เปรียบเทียบ 2 รูปพร้อมกราฟ winner (`compare_food_images`)

---

## 🗂️ โครงสร้างโปรเจกต์

```
MachineVision/
├── Dataset_for_development/
│   ├── Questionair Images/       # รูปภาพจาก Questionnaire
│   ├── Instagram Photos/         # รูปภาพจาก Instagram
│   ├── data_from_questionaire.csv
│   └── data_from_intragram.csv
├── food_classification/
│   ├── train/
│   │   ├── attractive/
│   │   └── unattractive/
│   └── val/
│       ├── attractive/
│       └── unattractive/
├── models/
│   ├── food_attractiveness_v2_final.keras
│   ├── class_names.json
│   └── model_config.json
├── food_attractiveness_v3_1.ipynb
└── README.md
```

---

## 📊 แหล่งข้อมูล

| แหล่งข้อมูล | จำนวน | น้ำหนัก | หมายเหตุ |
|---|---|---|---|
| 📋 Questionnaire | 500 คู่ (129 คนโหวต) | 1.0 | น้ำหนักสูงสุด |
| 📸 Instagram | 265 คู่ (5 หมวดอาหาร) | 0.4 | Burger, Dessert, Pizza, Ramen, Sushi |
| 🤖 Auto-label | ~21,904 รูป | 0.3 | Label อัตโนมัติด้วย Aesthetic Score |

---

## 🤖 Model Architecture

| | v2 | v3 |
|---|---|---|
| Backbone | EfficientNetV2S | **EfficientNetV2M** |
| Input size | 224px | **260px** |
| Data Augmentation | ❌ | ✅ flip/rotate/zoom/brightness |
| LR Schedule | Constant | **Cosine Decay** |
| Label Smoothing | ❌ | ✅ 0.1 |
| Batch size | 16 | **32** |
| Epochs (Phase1/Phase2) | 30/20 | **50/30** |

---

## ⚙️ Requirements

- Python **3.10 – 3.12** (TensorFlow ยังไม่รองรับ 3.13+)
- GPU (แนะนำ) หรือ CPU ก็ได้

### ติดตั้ง

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "tensorflow[and-cuda]" scikit-learn matplotlib seaborn pandas numpy tqdm streamlit
```

---

## 🚀 วิธีใช้งาน

### 1. Import และโหลดโมเดล

```python
import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

BASE_PATH  = '/your/path/to/MachineVision'
IMG_SIZE   = 260

MODEL_PATH = os.path.join(BASE_PATH, 'models', 'food_attractiveness_v2_final.keras')
CLASS_JSON = os.path.join(BASE_PATH, 'models', 'class_names.json')

loaded_model = keras.models.load_model(MODEL_PATH)
with open(CLASS_JSON) as f:
    CLASS_NAMES = json.load(f)
```

### 2. ทำนายรูปเดี่ยว

```python
result = predict_attractiveness('path/to/food.jpg', model=loaded_model)
print(result)
# {'label': 'attractive', 'confidence': 0.87, 'all_probs': {...}}
```

### 3. เปรียบเทียบ 2 รูป

```python
r1, r2 = compare_food_images('food1.jpg', 'food2.jpg', model=loaded_model)
```

ผลลัพธ์จะแสดง 3 panel: รูป 1, รูป 2, และ bar chart เปรียบเทียบ score พร้อมประกาศ 🏆 Winner

---

## 📈 Training Pipeline

| Step | รายละเอียด |
|---|---|
| Phase 1 | Freeze backbone — train classifier head เท่านั้น |
| Phase 2 | Unfreeze backbone บางส่วน — fine-tune ทั้งโมเดล |
| Loss | Categorical Crossentropy + Label Smoothing 0.1 |
| Optimizer | Adam + Cosine Decay LR |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |

---

## 📝 หมายเหตุ

- หลังจาก WSL restart ต้อง import library และ define ฟังก์ชันใหม่ทุกครั้ง (โมเดลที่ save ไว้ใน disk ไม่หาย)
- `class_names.json` สำคัญมาก — อย่าลบหรือเปลี่ยนชื่อ
- ถ้าเปลี่ยนชื่อไฟล์ `.keras` ต้องแก้ `MODEL_PATH` ตอน load ให้ตรงกันด้วย
