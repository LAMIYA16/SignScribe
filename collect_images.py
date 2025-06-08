import cv2
import os

gesture = input("Enter gesture name (hi, best_of_luck, love): ").strip()
save_path = f'dataset/{gesture}'
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror the image
    cv2.putText(frame, f"Images Saved: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Capture - Press 's' to save", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        filename = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
