# prompted with Chat GPT
import cv2
import numpy as np

# -------- paths --------
video_path = "/Users/raine/Desktop/X2.mov"      # input video
output_path = "/Users/raine/Desktop/X2_density_heatmap.mp4"  # output video

# heatmap resolution (internal)
heatmap_w, heatmap_h = 900, 900

# how long traces stay on screen (0–1, closer to 1 = more memory)
decay = 0.95

# how strong each point contributes to the heatmap
presence_strength = 20.0

# how big each point appears (in pixels on heatmap grid)
point_radius = 20

# gaussian blur to smooth the heatmap
blur_sigma = 35

# blend between heatmap and original frame
alpha_heat = 0.6   # heatmap weight
alpha_frame = 0.4  # original video weight


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video: {}".format(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

# heatmap buffer (float32 for accumulation)
heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

# kernel for expanding each point into a blob
kernel_size = (point_radius * 2 + 1, point_radius * 2 + 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # resize frame to heatmap resolution for easier processing
    small = cv2.resize(frame, (heatmap_w, heatmap_h))

    # detect green points in HSV space
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # decay previous heat slightly
    heatmap *= decay

    # expand green regions so each point becomes a blob
    expanded = cv2.dilate(mask, kernel)

    # add presence to heatmap wherever we see green
    heatmap[expanded > 0] += presence_strength

    # smooth heatmap so it looks like a continuous field
    heat_blur = cv2.GaussianBlur(heatmap, (0, 0), blur_sigma)

    # normalize to 0–255 for colormap
    heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX)
    heat_uint8 = heat_norm.astype(np.uint8)

    # apply color mapping (blue → red)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)

    # resize back to original frame size and overlay
    heat_resized = cv2.resize(heat_color, (frame_w, frame_h))
    overlay = cv2.addWeighted(heat_resized, alpha_heat, frame, alpha_frame, 0)

    out.write(overlay)

cap.release()
out.release()

print("saved density-based heatmap to:", output_path)
