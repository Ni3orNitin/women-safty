import os
import cv2
import time
import geocoder
import speech_recognition as sr
from twilio.rest import Client

# === Twilio Setup ===
account_sid = 'ACac74e42625e257922eab5cf7c2c92cd5'
auth_token = '44107377af70050145f36ab4b6b86d7b'
twilio_number = '+916207367883'
emergency_contact = '+916207367883'

# === AI Face Detection Model ===
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# === Directory to Save Images ===
save_dir = "captured_faces"
os.makedirs(save_dir, exist_ok=True)

# === Voice Command Listener ===
def listen_for_help():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🔊 Voice trigger is listening... Say 'help' or 'help me'")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"🎙️ You said: {command}")
                if "help" in command:
                    print("🚨 Emergency command detected!")
                    return True
            except sr.WaitTimeoutError:
                print("⏳ Listening timeout, retrying...")
            except sr.UnknownValueError:
                print("🤷 Couldn't understand. Listening again...")
            except sr.RequestError as e:
                print(f"❌ API error: {e}")
                return False

# === Get Device Location ===
def get_location():
    location = geocoder.ip('me')
    if location.ok:
        print(f"📍 Location fetched: {location.latlng}")
        return location.latlng
    print("⚠️ Could not fetch location.")
    return None

# === Send Emergency SMS ===
def send_alert(location):
    client = Client(account_sid, auth_token)
    lat, lon = location
    map_link = f"https://maps.google.com/?q={lat},{lon}"
    message = client.messages.create(
        body=f"🚨 Emergency! I need help.\nMy location: {map_link}",
        from_=twilio_number,
        to=emergency_contact
    )
    print(f"✅ Emergency SMS sent! SID: {message.sid}")

# === AI Face Capture ===
def start_face_capture():
    cam = cv2.VideoCapture(0)
    image_count = 0
    save_interval = 2  # seconds between saves
    last_saved = time.time()

    print("📸 Capturing faces... Press 'q' to stop.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("❌ Failed to read camera feed.")
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0), False, False)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    x1, y1, x2, y2 = box.astype("int")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    current_time = time.time()
                    if current_time - last_saved >= save_interval:
                        face_img = frame[y1:y2, x1:x2]
                        img_path = os.path.join(save_dir, f"face_{image_count}.jpg")
                        cv2.imwrite(img_path, face_img)
                        print(f"💾 Saved: {img_path}")
                        image_count += 1
                        last_saved = current_time

            cv2.imshow("AI Face Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Capture stopped by user (Ctrl+C).")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print(f"✅ Done. Total faces saved: {image_count}")

# === MAIN FLOW ===
if __name__ == "__main__":
    print("🛡️ Women Safety System is ACTIVE")

    if listen_for_help():
        location = get_location()
        if location:
            send_alert(location)
        start_face_capture()
    else:
        print("👍 No emergency detected.")


        