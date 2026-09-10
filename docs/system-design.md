# System design notes for boundary detection

Companion to the root README. These are the principles you apply after you can run a demo.

## 1. Contract before model

Write a one-page contract:

```text
Input:     RGB | RGB-D | 16-bit medical volume
Resolution: native capture, not "whatever the paper used"
Latency:   p50 / p95 end-to-end, including IO
Output:    mask | polygon | polyline | grasp pose
Quality:   metric + operating point + cost of FN vs FP
Safety:    what happens when confidence < τ
```

If two stakeholders disagree on the contract, no architecture will save the project.

## 2. Reference architecture

```
sensor
  → sync + timestamp
  → undistort / resample
  → ROI gate
  → cheap proposal (edge | detector | tracker)
  → expensive refine (instance head | SAM)
  → geometry fit
  → decision + overlay
  → actuator / archive
```

Keep the cheap path always-on. Use the expensive path only when the cheap path is uncertain or when the object is new.

## 3. Data units that do not lie

Store, in one record per frame:

| Field | Unit / type |
| --- | --- |
| timestamp | UTC ns |
| camera_id | string |
| image_size | (H, W) px |
| mask | HxW uint8 or RLE |
| polygon | list[(x_px, y_px)] in image frame |
| score | [0, 1] calibrated |
| prompt | box / point / text that produced the mask |
| model_id | name + commit + weights hash |

Without `prompt` and `model_id`, foundation-model systems are not reproducible.

## 4. Boundary-aware losses and metrics

mIoU is necessary and insufficient.

Useful extras:

- Boundary F-score / trimap IoU
- Hausdorff distance (lesions)
- clDice / skeleton recall (cracks, vessels)
- ODS / OIS (classic edge detection on BSDS)
- Panoptic PQ when instances matter
- Grasp success @ k attempts when robotics is the product

## 5. Failure budget

Allocate error before you train:

```
total allowable miss rate = 1%
  sensor / focus          0.2%
  domain shift            0.3%
  model                   0.3%
  postprocess / NMS       0.1%
  operator overlay lag    0.1%
```

If the camera cannot resolve a 0.2 mm crack, a transformer will not invent it.

## 6. When to use which layer

| Situation | First tool |
| --- | --- |
| Controlled lighting, high contrast | Canny + morphology |
| Need class labels | semantic seg + boundary loss |
| Objects touch | instance seg |
| Thin dark lines | high-res U-Net / DeepCrack |
| No labels, operator in the loop | SAM / SAM 2 |
| Text query ("the cracked box") | Grounded-SAM |
| 30+ FPS on an edge box | YOLO-seg or PiDiNet |
| Clinical volume | nnU-Net, not SAM out of the box |

## 7. Safety and governance

- Do not close a robot gripper on an unvalidated mask.
- Do not auto-diagnose from a lesion contour.
- Factory and hospital images are usually not redistributable.
- Prefer on-prem weights when the scene is confidential.
- Keep a golden set of 50–200 frames that you never train on and that you replay after every model bump.

## 8. Minimal production test list

1. Empty scene — no spurious contours.
2. One isolated object — polygon area within 5%.
3. Two touching objects — two instances, not one.
4. Motion blur — score drops instead of silently shrinking.
5. Lighting step — decision does not flip at the frame boundary.
6. Known defect smaller than 3 px — either detected or explicitly “below resolution”.
