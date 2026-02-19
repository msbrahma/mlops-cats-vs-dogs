from fastapi import FastAPI, UploadFile, File, Request
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import time
import logging
from src.model import SimpleCNN

app = FastAPI()

# -----------------------
# Basic Monitoring Setup
# -----------------------
request_count = 0
total_latency = 0.0

logging.basicConfig(level=logging.INFO)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count, total_latency

    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    request_count += 1
    total_latency += latency

    logging.info(f"{request.method} {request.url.path} "
                 f"Status: {response.status_code} "
                 f"Latency: {latency:.4f}s")

    return response

@app.get("/metrics")
def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0
    return {
        "total_requests": request_count,
        "average_latency": round(avg_latency, 4)
    }

# -----------------------
# Model Setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN()
model.load_state_dict(torch.load("models/cnn_model.pt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = "Cat" if predicted.item() == 0 else "Dog"

    return {
        "prediction": label,
        "confidence": float(confidence.item())
    }
