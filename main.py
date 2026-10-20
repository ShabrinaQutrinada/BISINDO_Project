import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Webcam tidak dapat dibaca.")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("BISINDO Project - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()