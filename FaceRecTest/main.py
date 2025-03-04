import cv2

# Load the face detection classifier (why xml)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around faces
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (15, 15, 15), 2)

    # Display the output
    cv2.imshow("Face Detection", frame)

    # List number of faces detected and informational

    face_count = len(faces)

    cv2.putText(
        frame,
        f"Press 'q' to quit, faces = {face_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Face Detection", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
