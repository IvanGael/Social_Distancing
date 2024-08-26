import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from scipy.spatial import distance as dist

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load MiDaS model for depth estimation
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_model.eval()

midas_transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set distance threshold (in pixels) for social distancing
MIN_DISTANCE = 50

RED_COLOR = (121, 28, 248)
GREEN_COLOR = (194, 247, 50)

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = midas_transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = midas_model(img)

    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.cpu().numpy()
    return depth

def draw_boxes_and_lines(frame, boxes, violations):
    person_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if i in violations:
            color = RED_COLOR  # Red for violations
        else:
            color = GREEN_COLOR  # Green for compliance
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.rectangle(person_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    
    for combination in violations:
        idx1, idx2 = combination
        centroid1 = get_centroid(boxes[idx1])
        centroid2 = get_centroid(boxes[idx2])
        cv2.line(frame, centroid1, centroid2, RED_COLOR, 2)
    
    return person_mask

def get_centroid(box):
    return (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))

def draw_social_distancing_map(frame, boxes, violations):
    map_size = 250
    map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    
    for i, box in enumerate(boxes):
        centroid = get_centroid(box)
        normalized_x = int(centroid[0] * map_size / frame.shape[1])
        normalized_y = int(centroid[1] * map_size / frame.shape[0])
        
        color = RED_COLOR if i in set([item for sublist in violations for item in sublist]) else GREEN_COLOR
        cv2.circle(map_img, (normalized_x, normalized_y), 5, color, -1)
    
    return map_img

cap = cv2.VideoCapture("video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while True:
    success, frame = cap.read()
    if not success:
        break

    # Estimate depth
    depth = estimate_depth(frame)

    results = model(frame)
    boxes = results.xyxy[0].cpu().numpy()
    
    # Filter for person class (class 0 in COCO dataset)
    boxes = boxes[boxes[:, 5] == 0][:, :4]
    
    violations = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            centroid1 = get_centroid(boxes[i])
            centroid2 = get_centroid(boxes[j])
            if dist.euclidean(centroid1, centroid2) < MIN_DISTANCE:
                violations.append((i, j))
    
    # Draw boxes and get person mask
    person_mask = draw_boxes_and_lines(frame, boxes, violations)
    
    map_img = draw_social_distancing_map(frame, boxes, violations)
    
    # Create depth colormap
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    
    # Apply depth map only to person areas
    depth_overlay = np.zeros_like(frame)
    depth_overlay[person_mask == 255] = depth_colormap[person_mask == 255]
    frame = cv2.addWeighted(frame, 1, depth_overlay, 0.3, 0)

    # Add the map to the top-right corner of the frame
    frame[-map_img.shape[0]:, -map_img.shape[1]:] = map_img
    
    cv2.putText(frame, f"SOCIAL DISTANCING", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 3)
    cv2.putText(frame, f"VIOLATION : {len(violations)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 3)
    
    # Write the frame to the output video file
    out.write(frame)

    cv2.namedWindow('Social Distancing Monitoring', cv2.WINDOW_NORMAL)
    cv2.imshow('Social Distancing Monitoring', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()