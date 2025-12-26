import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import setup_path
from train_model import SimpleCNN
from utils.constants import CLASSES

# -------------------
# Config
# -------------------
IMG_SIZE = 100
CONF_THRESHOLD = 0.6

FRUITNET_MODEL = "results/fruitnet_cnn.pth"
FRUITS360_MODEL = "results/fruits360_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ROI size (relative to frame)
ROI_SCALE = 0.5  # 50% of frame width/height

# -------------------
# Load models
# -------------------
fruitnet_model = SimpleCNN().to(device)
fruitnet_model.load_state_dict(torch.load(FRUITNET_MODEL, map_location=device))
fruitnet_model.eval()

fruits360_model = SimpleCNN().to(device)
fruits360_model.load_state_dict(torch.load(FRUITS360_MODEL, map_location=device))
fruits360_model.eval()

# -------------------
# Transform
# -------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# -------------------
# Webcam
# -------------------
cap = cv2.VideoCapture(1)     # change index to select appropriate webcam
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Live comparison with ROI cropping. Place fruit inside the box. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define central ROI
    roi_w = int(w * ROI_SCALE)
    roi_h = int(h * ROI_SCALE)
    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Crop ROI
    roi = frame[y1:y2, x1:x2]
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    input_tensor = transform(rgb_roi).unsqueeze(0).to(device)

    with torch.no_grad():
        # FruitNet
        out_fn = fruitnet_model(input_tensor)
        probs_fn = F.softmax(out_fn, dim=1)
        conf_fn, pred_fn = probs_fn.max(dim=1)

        # Fruits-360
        out_f360 = fruits360_model(input_tensor)
        probs_f360 = F.softmax(out_f360, dim=1)
        conf_f360, pred_f360 = probs_f360.max(dim=1)

    def format_prediction(pred, conf):
        if conf < CONF_THRESHOLD:
            return "Unsure", (0, 0, 255)
        return f"{CLASSES[pred]} ({conf:.2f})", (0, 255, 0)

    fn_text, fn_color = format_prediction(pred_fn.item(), conf_fn.item())
    f360_text, f360_color = format_prediction(pred_f360.item(), conf_f360.item())

    cv2.putText(frame, f"FruitNet: {fn_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, fn_color, 2)

    cv2.putText(frame, f"Fruits-360: {f360_text}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, f360_color, 2)

    cv2.imshow("Live Dataset Comparison (ROI)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
