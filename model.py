# %%
from ultralytics import YOLO

# %%
# Model Initialization
model = YOLO("yolov8n.pt")

# Model Training
model.train(data="data.yaml", epochs=20, imgsz=640)

# %%
results = model("mountain.jpg")
print(results)