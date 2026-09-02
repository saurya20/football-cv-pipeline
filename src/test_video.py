from ultralytics import YOLO

VIDEO_PATH = "data/videos/game1.mp4"

model = YOLO("yolo11n.pt")

results = model.track(
    source=VIDEO_PATH,
    tracker="botsort.yaml",
    conf=0.3,
    save=True,
    project="outputs",
    name="nepal_tracking",
    stream=True
)

for result in results:
    pass

print("Video processing complete.")