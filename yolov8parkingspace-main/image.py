import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLO
model = YOLO('yolo11m.pt')  # คุณสามารถเปลี่ยนไปใช้โมเดลอื่นได้ 

# โหลดรูปภาพ
image_path = 'bb.jpg'  # ใส่ชื่อไฟล์รูปภาพที่คุณต้องการตรวจจับ
image = cv2.imread(image_path)
image = cv2.resize(image, (1020, 500))
# ทำนายวัตถุในภาพ
results = model.predict(image,conf=0.75)

# ดึงข้อมูลผลลัพธ์
boxes = results[0].boxes.data  # Bounding boxes ที่ตรวจจับได้
for box in boxes:
    x1, y1, x2, y2, conf, cls_id = map(int, box[:6])
    class_name = model.names[cls_id]  # ชื่อคลาส (เช่น car, person)

    # กรองเฉพาะคลาส "car"
    if class_name == 'car':
        # วาดกรอบรอบรถ
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # ใส่ชื่อคลาสและความมั่นใจ
        cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# แสดงผลลัพธ์
cv2.imshow("Detected Cars", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
