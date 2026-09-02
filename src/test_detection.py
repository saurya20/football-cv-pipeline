from ultralytics import YOLO
from pathlib import Path


# -----------------------------
# SETTINGS
# -----------------------------

FRAME_DIR = Path("data/Eliteserien/2021/frames/2845")
LABEL_DIR = Path("data/Eliteserien/2021/detection/2845")

MODEL_PATH = "yolo11n.pt"

# IoU threshold:
# A prediction counts as correct if it overlaps
# a ground-truth box by at least 50%.
IOU_THRESHOLD = 0.5


# -----------------------------
# HELPER: CALCULATE IoU
# -----------------------------

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes.

    Boxes are:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


# -----------------------------
# CONVERT YOLO LABEL TO PIXELS
# -----------------------------

def yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height):

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [x1, y1, x2, y2]


# -----------------------------
# LOAD MODEL
# -----------------------------

model = YOLO(MODEL_PATH)


# -----------------------------
# TOTAL METRICS
# -----------------------------

total_tp = 0
total_fp = 0
total_fn = 0


# -----------------------------
# PROCESS EACH FRAME
# -----------------------------

frames = sorted(FRAME_DIR.glob("*.jpg"))

print(f"Evaluating {len(frames)} frames...")
print()


for frame_path in frames:

    label_path = LABEL_DIR / f"{frame_path.stem}.txt"

    # Run YOLO
    results = model(str(frame_path), verbose=False)

    result = results[0]

    image_height, image_width = result.orig_shape

    # ---------------------------------
    # YOLO predictions
    # ---------------------------------

    predicted_boxes = []

    for box in result.boxes:

        # COCO class 0 = person
        class_id = int(box.cls[0])

        if class_id != 0:
            continue

        confidence = float(box.conf[0])

        # Only keep reasonably confident detections
        if confidence < 0.3:
            continue

        coordinates = box.xyxy[0].tolist()

        predicted_boxes.append(coordinates)

    # ---------------------------------
    # Ground truth
    # ---------------------------------

    ground_truth_boxes = []

    if label_path.exists():

        with open(label_path, "r") as file:

            for line in file:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])

                # SoccerSum:
                # 0 = Player
                # 1 = Goalkeeper
                #
                # We only evaluate players + goalkeepers.

                if class_id not in [0, 1]:
                    continue

                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

                box = yolo_to_xyxy(
                    x_center,
                    y_center,
                    width,
                    height,
                    image_width,
                    image_height
                )

                ground_truth_boxes.append(box)

    # ---------------------------------
    # MATCH PREDICTIONS TO GROUND TRUTH
    # ---------------------------------

    matched_ground_truth = set()

    true_positives = 0
    false_positives = 0

    for prediction in predicted_boxes:

        best_iou = 0
        best_index = None

        for i, ground_truth in enumerate(ground_truth_boxes):

            if i in matched_ground_truth:
                continue

            iou = calculate_iou(prediction, ground_truth)

            if iou > best_iou:

                best_iou = iou
                best_index = i

        if best_iou >= IOU_THRESHOLD:

            true_positives += 1
            matched_ground_truth.add(best_index)

        else:

            false_positives += 1

    false_negatives = len(ground_truth_boxes) - len(matched_ground_truth)

    total_tp += true_positives
    total_fp += false_positives
    total_fn += false_negatives

    print(
        f"{frame_path.name}: "
        f"GT={len(ground_truth_boxes)} "
        f"Pred={len(predicted_boxes)} "
        f"TP={true_positives} "
        f"FP={false_positives} "
        f"FN={false_negatives}"
    )


# -----------------------------
# FINAL METRICS
# -----------------------------

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)


# -----------------------------
# RESULTS
# -----------------------------

print()
print("=" * 50)
print("DETECTION EVALUATION")
print("=" * 50)

print(f"Frames evaluated : {len(frames)}")
print(f"True positives   : {total_tp}")
print(f"False positives  : {total_fp}")
print(f"False negatives  : {total_fn}")

print()
print(f"Precision        : {precision:.3f}")
print(f"Recall           : {recall:.3f}")
print(f"F1 Score         : {f1:.3f}")

print("=" * 50)