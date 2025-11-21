from ultralytics import YOLO
import pyttsx3

# Load trained best model
model = YOLO(r"C:\Users\Kavipriya\Downloads\safety_dataset\runs\detect\train4\weights\best.pt")

# Text-to-speech
engine = pyttsx3.init()

# Test image (img3)
results = model(r"C:\Users\Kavipriya\Downloads\safety_dataset\images\val\img3.jpg")

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])

        if cls == 0:
            text = "Helmet detected"
        elif cls == 1:
            text = "No helmet detected"
        elif cls == 2:
            text = "Face detected"
        else:
            text = "Unknown object"

        print(text)
        engine.say(text)

engine.runAndWait()