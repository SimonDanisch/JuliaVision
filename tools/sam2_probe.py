"""First contact with SAM 2.1: one click on a frame -> mask, measured.

Answers the question the matte tool is stuck on — whether a click gives an
accurate object mask — before any of it is ported. Coverage is reported the way
`VideoEditor.mattecoverage` reports it, so the numbers are comparable to the
ones measured for painted seeds.
"""
import sys, time, numpy as np, torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

frame = np.array(Image.open(sys.argv[1]).convert("RGB"))
ckpt, cfg = sys.argv[2], sys.argv[3]
px, py = float(sys.argv[4]), float(sys.argv[5])          # normalized click
H, W = frame.shape[:2]

t0 = time.time()
model = build_sam2(cfg, ckpt, device="cuda")
pred = SAM2ImagePredictor(model)
print(f"load {time.time()-t0:5.1f}s  frame {W}x{H}", flush=True)

t0 = time.time(); pred.set_image(frame); t_embed = time.time()-t0
pt = np.array([[px*W, py*H]]); lb = np.array([1])
t0 = time.time()
masks, scores, _ = pred.predict(point_coords=pt, point_labels=lb, multimask_output=True)
t_pred = time.time()-t0
print(f"embed {t_embed:5.2f}s   predict {t_pred:5.3f}s", flush=True)
for i, (m, s) in enumerate(zip(masks, scores)):
    print(f"  mask {i}: score {s:.3f}  covers {100*m.mean():5.1f}% of the frame")
best = int(np.argmax(scores))
Image.fromarray((masks[best]*255).astype(np.uint8)).save("/tmp/sam2_mask.png")
print("best ->", best, "saved /tmp/sam2_mask.png")
