# MachineVision (Food Aesthetic Pairwise)

โปรเจกต์นี้ใช้โมเดล Siamese เปรียบเทียบว่า "รูปอาหาร A หรือ B ดูน่ากินกว่า"

## Quick Start

1. เข้าโฟลเดอร์โปรเจกต์

```bash
cd /home/besstsu/Documents/MachineVision
```

2. สร้าง virtual environment (แนะนำใช้ `.venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. รันโหมดเปรียบเทียบแบบ interactive

```bash
python Food_choser.py
```

4. หรือรันแบบส่ง path รูปผ่าน argument

```bash
python Food_choser.py \
  --image-a ./dataset_split_IG/val/images/Burger/320058631_1106504873331527_1104522570667449301_n.jpg \
  --image-b ./dataset_split_IG/val/images/Pizza/321174604_542695657877364_2475541234582595236_n.jpg
```

## Train Model

```bash
python trainer.py
```

ไฟล์โมเดลผลลัพธ์จะถูกบันทึกเป็น `food_aesthetic_v7_master.keras`

## Notes

- ต้องรันจาก environment ที่มี TensorFlow (`.venv` ตามตัวอย่าง)
- ถ้าเคยใช้ `venv/` เดิมแล้วขึ้น `ModuleNotFoundError: tensorflow` ให้สลับมาใช้ `.venv` หรือ install ใหม่ตาม `requirements.txt`
- `Food_choser.py` โหลดโมเดลจาก `food_aesthetic_v7_master.keras` โดยอิง path ตามตำแหน่งไฟล์อัตโนมัติ

## Before RUN
ขั้นตอนเทรนใหม่ทุกครั้ง
1. เปิด VS Code จาก WSL Terminal
bashcd ~/Documents/MachineVision
source .venv/bin/activate
code .
2. เลือก Kernel ให้ถูก

มุมบนขวาของ notebook → "Python 3 (venv) (Python 3.12.3)"

3. รัน Cell ตามลำดับ

Ctrl+Shift+P → "Restart Kernel and Run All Cells" — รันทีเดียวจบเลย


ถ้าอยากเทรนใหม่ตั้งแต่ต้น (ล้างของเก่า)
bash# ลบโมเดลและ dataset split เก่าออก
rm -rf ~/Documents/MachineVision/models_v2
rm -rf ~/Documents/MachineVision/food_classification_v2
แล้วรัน notebook ใหม่ทั้งหมด