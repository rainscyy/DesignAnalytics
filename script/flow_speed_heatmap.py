# prompted with Chat GPT
import cv2
import numpy as np

# -------- paths --------
video_path = "/Users/raine/Desktop/X2.mov"
output_path = "/Users/raine/Desktop/X2_flow_speed_heatmap.mp4"

# internal heatmap resolution (can be lower than original)
heatmap_w, heatmap_h = 900, 900

# how much history to keep (0–1)
decay = 0.94

# scale factor for how much each frame's motion contributes
motion_strength = 4.0

# ignore very small motions (noise threshold in pixels/frame)
motion_threshold = 0.5

# smoothing of the heat field
blur_sigma = 25

# blending with original video
alpha_heat = 0.6
alpha_frame = 0.4


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video: {}".format(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Could not read first frame from video.")

# resize first frame to heatmap space and convert to gray
prev_small = cv2.resize(first_frame, (heatmap_w, heatmap_h))
prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # work at heatmap resolution
    small = cv2.resize(frame, (heatmap_w, heatmap_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # dense optical flow between previous and current frame
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)

    # drop tiny motions to reduce noise
    mag_filtered = mag.copy()
    mag_filtered[mag < motion_threshold] = 0.0

    # update heatmap: decay old values, add motion from this frame
    heatmap *= decay
    heatmap += mag_filtered * motion_strength

    # smooth the heatmap
    heat_blur = cv2.GaussianBlur(heatmap, (0, 0), blur_sigma)

    # normalize and colorize
    heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX)
    heat_uint8 = heat_norm.astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)

    # blend with original frame
    heat_resized = cv2.resize(heat_color, (frame_w, frame_h))
    overlay = cv2.addWeighted(heat_resized, alpha_heat, frame, alpha_frame, 0)

    out.write(overlay)

    prev_gray = gray

cap.release()
out.release()

print("saved flow-based speed heatmap to:", output_path)
