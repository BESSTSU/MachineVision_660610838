"""
🍣 Food Attractiveness Rater — Streamlit App
รัน: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
import io

# ── ตั้งค่าหน้าเว็บ ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍣 Food Attractiveness Rater",
    page_icon="🍣",
    layout="wide",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .winner-box {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
        color: white;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .score-bar-wrap {
        background: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        height: 28px;
        margin: 6px 0;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        color: white;
        font-weight: bold;
        font-size: 14px;
        transition: width 0.5s ease;
    }
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ── Path Config ──────────────────────────────────────────────────────────────
BASE_PATH  = '/home/besstsu/Documents/MachineVision'
MODEL_DIR  = os.path.join(BASE_PATH, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'food_attractiveness_v3_1.keras')
CLASS_JSON = os.path.join(MODEL_DIR, 'class_names.json')
IMG_SIZE   = 224

# ── โหลดโมเดล (cache ไว้ไม่โหลดซ้ำ) ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON) as f:
        class_names = json.load(f)
    return model, class_names

# ── ฟังก์ชัน Predict ─────────────────────────────────────────────────────────
def predict(image_pil, model, class_names, img_size=IMG_SIZE):
    """รับ PIL Image → คืน dict {label, confidence, score_attractive}"""
    img = image_pil.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "label"      : class_names[idx],
        "confidence" : float(probs[idx]),
        "all_probs"  : {c: float(probs[i]) for i, c in enumerate(class_names)},
        "score"      : float(probs[class_names.index("attractive")]),
    }

def score_bar(score: float, color: str, label: str):
    pct = int(score * 100)
    st.markdown(f"""
    <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:{pct}%; background:{color};">
            {pct}%
        </div>
    </div>
    <div style="text-align:center; font-size:13px; color:#555;">{label}</div>
    """, unsafe_allow_html=True)

# ── หน้าหลัก ────────────────────────────────────────────────────────────────
st.title("🍣 Food Attractiveness Rater")
st.caption("อัปโหลดรูปอาหาร 2 รูป แล้วให้ AI ตัดสินว่าอันไหนน่ากินกว่า!")

# โหลดโมเดล
model, class_names = load_model()

if model is None:
    st.error(f"""
    ❌ ไม่พบโมเดล กรุณาเทรนโมเดลให้เสร็จก่อน  
    โมเดลควรอยู่ที่: `{MODEL_PATH}`
    """)
    st.info("รัน notebook `food_attractiveness_v3.ipynb` ให้ครบก่อนใช้งาน app นี้ครับ")
    st.stop()

st.success(f"✅ โมเดลพร้อมใช้งาน | Classes: {class_names}")

st.divider()

# ── อัปโหลดรูป ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ รูปที่ 1")
    f1 = st.file_uploader("เลือกรูปอาหาร", type=["jpg","jpeg","png"], key="img1")

with col2:
    st.subheader("🖼️ รูปที่ 2")
    f2 = st.file_uploader("เลือกรูปอาหาร", type=["jpg","jpeg","png"], key="img2")

# ── ประมวลผล ─────────────────────────────────────────────────────────────────
if f1 and f2:
    img1 = Image.open(f1)
    img2 = Image.open(f2)

    # แสดงรูป
    with col1:
        st.image(img1, use_container_width=True)
    with col2:
        st.image(img2, use_container_width=True)

    st.divider()

    # Predict
    with st.spinner("🤔 AI กำลังวิเคราะห์..."):
        r1 = predict(img1, model, class_names)
        r2 = predict(img2, model, class_names)

    s1, s2 = r1["score"], r2["score"]
    winner_idx = 1 if s1 >= s2 else 2
    diff = abs(s1 - s2)

    # ── แสดงผลลัพธ์ ──────────────────────────────────────────────────────────
    st.subheader("📊 ผลการวิเคราะห์")

    # Winner banner
    margin_text = (
        "สูสีมาก 🤏" if diff < 0.05 else
        "ชนะนิดหน่อย" if diff < 0.15 else
        "ชนะชัดเจน ✨" if diff < 0.30 else
        "ชนะขาดลอย 🏆"
    )
    st.markdown(f"""
    <div class="winner-box">
        🏆 รูปที่ {winner_idx} น่ากินกว่า! — {margin_text}
    </div>
    """, unsafe_allow_html=True)

    # Score bars
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**รูปที่ 1**")
        color1 = "#4CAF50" if winner_idx == 1 else "#9E9E9E"
        score_bar(s1, color1, f"Attractiveness Score: {s1:.3f}")

    with col_s2:
        st.markdown("**รูปที่ 2**")
        color2 = "#4CAF50" if winner_idx == 2 else "#9E9E9E"
        score_bar(s2, color2, f"Attractiveness Score: {s2:.3f}")

    # Detail expander
    with st.expander("🔍 ดูรายละเอียด"):
        dc1, dc2 = st.columns(2)
        with dc1:
            st.write("**รูปที่ 1**")
            for k, v in r1["all_probs"].items():
                st.write(f"- {k}: `{v:.4f}`")
        with dc2:
            st.write("**รูปที่ 2**")
            for k, v in r2["all_probs"].items():
                st.write(f"- {k}: `{v:.4f}`")

elif f1 or f2:
    st.info("📎 กรุณาอัปโหลดรูปทั้ง **2 รูป** เพื่อเปรียบเทียบครับ")
else:
    # Placeholder
    st.markdown("""
    <div style="text-align:center; padding:60px; color:#aaa; font-size:18px;">
        ⬆️ อัปโหลดรูปอาหาร 2 รูปด้านบน<br>แล้ว AI จะตัดสินว่าอันไหนน่ากินกว่า
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ข้อมูลโมเดล")
    config_path = os.path.join(MODEL_DIR, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        st.json({
            "backbone"       : cfg.get("backbone"),
            "img_size"       : cfg.get("img_size"),
            "best_val_acc"   : f"{cfg.get('best_val_accuracy', 0):.2%}",
            "train_samples"  : cfg.get("n_train"),
        })
    else:
        st.info("model_config.json ยังไม่มี")

    st.divider()
    st.markdown("""
    **วิธีใช้:**
    1. อัปโหลดรูปอาหาร 2 รูป
    2. AI จะวิเคราะห์อัตโนมัติ
    3. ดูผลว่าอันไหนน่ากินกว่า
    
    **รูปที่ดีควร:**
    - ชัด ไม่เบลอ
    - แสงดี ไม่มืด/สว่างเกิน
    - เห็นอาหารชัดเจน
    """)
