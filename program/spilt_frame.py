import cv2
import os

video_path = r"D:\license_plate_and_car_detection\video\video_2.mp4"
output_folder = "night_frame"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

# อ่านค่า FPS ของวิดีโอ
fps = cap.get(cv2.CAP_PROP_FPS)

# ===== กำหนดเวลาที่ต้องการเซฟ =====
save_every = 0.5      # เปลี่ยนเป็น 1 ถ้าต้องการทุก 1 วินาที

# คำนวณว่าเท่ากับกี่เฟรม
interval = int(fps * save_every)

frame_count = 0
image_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ทุก ๆ interval เฟรม ให้เซฟรูป
    if frame_count % interval == 0:
        filename = os.path.join(output_folder, f"frame3_{image_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        image_count += 1

    frame_count += 1

cap.release()

print(f"Saved {image_count} images.")