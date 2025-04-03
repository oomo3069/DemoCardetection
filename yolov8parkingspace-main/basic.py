import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLO
model = YOLO('yolo11s.pt')

# โหลด class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# อ่านวิดีโอ
cap = cv2.VideoCapture('video_20250126_160901.mp4')

# ตัวแปรเก็บพิกัดช่องจอด
parking_areas = []
current_area = []

# ฟังก์ชันรับค่าพิกัดจากการคลิกเมาส์
def select_parking_spots(event, x, y, flags, param):
    global current_area, parking_areas

    if event == cv2.EVENT_LBUTTONDOWN:  # คลิกซ้ายเพื่อกำหนดพิกัด
        current_area.append((x, y))
        print(f"Point selected: {x}, {y}")

    if len(current_area) == 4:  # เมื่อได้ครบ 4 จุด ให้เพิ่มเข้าไปใน list
        parking_areas.append(current_area)
        print(f"New Parking Area: {current_area}")
        current_area = []  # เคลียร์ค่าเพื่อสร้างช่องใหม่

# สร้างหน้าต่างและกำหนดฟังก์ชันคลิก
cv2.namedWindow("Select Parking Spots")
cv2.setMouseCallback("Select Parking Spots", select_parking_spots)

# แสดงเฟรมแรกเพื่อเลือกช่องจอด
ret, frame = cap.read()
if ret:
    frame = cv2.resize(frame, (1020, 500))
    while True:
        temp_frame = frame.copy()
        for area in parking_areas:
            cv2.polylines(temp_frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
        cv2.imshow("Select Parking Spots", temp_frame)

        if cv2.waitKey(1) & 0xFF == 13:  # กด Enter เพื่อเริ่มตรวจจับรถ
            break

cv2.destroyWindow("Select Parking Spots")

total_slots = len(parking_areas)  # จำนวนช่องจอดทั้งหมด

while True:    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame, conf=0.5)
    px = pd.DataFrame(results[0].boxes.data).astype("float")

    occupied_slots = 0  # จำนวนช่องที่ถูกใช้

    for i, area in enumerate(parking_areas):
        car_count = 0  # จำนวนรถในช่องนั้น

        for _, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row)
            c = class_list[d]

            if 'car' in c:  # ตรวจจับเฉพาะรถยนต์
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ตรวจสอบว่ารถอยู่ในช่องจอดหรือไม่
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    car_count += 1

        # นับจำนวนช่องจอดที่ถูกใช้
        if car_count > 0:
            occupied_slots += 1
            cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)  # สีแดงเมื่อเต็ม
        else:
            cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)  # สีเขียวเมื่อว่าง

    free_slots = total_slots - occupied_slots  # คำนวณที่จอดว่าง

    # แสดงจำนวนช่องว่างที่มุมซ้ายบน
    text_color = (0, 255, 0) if free_slots > 0 else (0, 0, 255)
    cv2.putText(frame, f"Free Slots: {free_slots}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    cv2.imshow("Parking Detection", frame)

    if cv2.waitKey(30) & 0xFF == 27:  # กด ESC เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()