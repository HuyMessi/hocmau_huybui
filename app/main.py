"""
GIAI ĐOẠN 5 - ROI + State machine + YOLO Detector -> Crop -> YOLO Classifier (6 labels)

Bản sửa dựa trên main.py A2 của bạn:
- Giữ state machine: NO_OBJECT / NOT_STABLE / VOTING / RESULT
- Detector (trash-object) trong ROI
- Khi detector đủ "CONF_LOCK" + ổn định -> vào VOTING
- Trong VOTING: crop bbox + chạy classifier -> vote nhãn
- Nếu vote đủ mạnh + conf classifier đủ -> RESULT

Cài:
  pip install opencv-python numpy ultralytics
"""

import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
from ultralytics import YOLO


# =========================
# LOAD MODELS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DETECTOR_PATH = PROJECT_ROOT / "models" / "detector" / "best" / "trash_object_best.pt"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "classifier" / "best" / "YOLO_best.pt" 

if not DETECTOR_PATH.exists():
    raise FileNotFoundError(f"Không thấy detector tại: {DETECTOR_PATH}")
if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(f"Không thấy classifier tại: {CLASSIFIER_PATH}")

detector = YOLO(str(DETECTOR_PATH))
classifier = YOLO(str(CLASSIFIER_PATH))

print("Loaded detector :", DETECTOR_PATH)
print("Loaded classifier:", CLASSIFIER_PATH)


# =========================
# CẤU HÌNH ROI + YOLO
# =========================
CAMERA_INDEX = 0

ROI_W_FRAC = 0.50
ROI_H_FRAC = 0.50

# ===== Device =====
# - Windows/Intel: "cpu"
DEVICE = "cpu"

# ===== Detector =====
YOLO_IMGSZ = 512
YOLO_CONF = 0.20
YOLO_IOU = 0.45

# Ngưỡng CHỐT detector (cao để chống tay/nhầm)
CONF_LOCK = 0.60  # ví dụ tay ~0.58 sẽ không vào VOTING/RESULT

# bbox phải đủ lớn
MIN_BOX_AREA_RATIO = 0.02

# ổn định (tâm bbox)
STABLE_SECONDS_REQUIRED = 0.4
MAX_CENTER_JITTER_PX = 18

# voting thời gian
VOTING_SECONDS = 0.8

# ===== Classifier =====
CLS_IMGSZ = 224              # thường 224 cho classifier
CLS_UNKNOWN_CONF = 0.55      # nếu top1 < ngưỡng => Unknown (giảm nhận sai)

# voting label
VOTE_DEQUE_MAXLEN = 20
VOTE_MIN_RATIO = 0.60        # nhãn thắng phải chiếm >=60% trong cửa sổ vote

# crop padding (lấy thêm bối cảnh)
CROP_PAD_FRAC = 0.15         # 15% bbox

FONT = cv2.FONT_HERSHEY_SIMPLEX


# =========================
# STATE MACHINE
# =========================
STATE_NO_OBJECT = "NO_OBJECT"
STATE_NOT_STABLE = "NOT_STABLE"
STATE_VOTING = "VOTING"
STATE_RESULT = "RESULT"


def compute_roi_rect(frame_w: int, frame_h: int):
    roi_w = int(frame_w * ROI_W_FRAC)
    roi_h = int(frame_h * ROI_H_FRAC)

    x1 = (frame_w - roi_w) // 2
    y1 = (frame_h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    return x1, y1, x2, y2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yolo_detect_in_roi(roi_bgr: np.ndarray):
    """
    Detect trong ROI.
    Trả về: has_obj, xyxy_roi, center_xy, conf, area_ratio, label
    """
    results = detector.predict(
        source=roi_bgr,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        device=DEVICE,
        verbose=False
    )

    r0 = results[0]
    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return False, None, None, 0.0, 0.0, ""

    confs = boxes.conf.detach().cpu().numpy()
    best_i = int(np.argmax(confs))

    xyxy = boxes.xyxy[best_i].detach().cpu().numpy().astype(int)
    x1, y1, x2, y2 = map(int, xyxy)
    conf = float(confs[best_i])

    roi_h, roi_w = roi_bgr.shape[:2]
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    roi_area = roi_w * roi_h
    area_ratio = float(box_area) / float(roi_area + 1e-9)

    cls_id = int(boxes.cls[best_i].detach().cpu().numpy())
    label = r0.names.get(cls_id, "trash-object") if hasattr(r0, "names") else "trash-object"

    # Gating theo kích thước bbox
    if area_ratio < MIN_BOX_AREA_RATIO:
        return False, None, None, conf, area_ratio, label

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return True, (x1, y1, x2, y2), (cx, cy), conf, area_ratio, label


def crop_with_padding(roi_bgr: np.ndarray, xyxy_roi, pad_frac=0.15):
    """Crop bbox trong ROI với padding."""
    h, w = roi_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy_roi
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)

    cx1 = clamp(x1 - pad_x, 0, w - 1)
    cy1 = clamp(y1 - pad_y, 0, h - 1)
    cx2 = clamp(x2 + pad_x, 0, w - 1)
    cy2 = clamp(y2 + pad_y, 0, h - 1)

    crop = roi_bgr[cy1:cy2, cx1:cx2].copy()
    return crop


def yolo_classify_crop(crop_bgr: np.ndarray):
    """
    YOLO classifier trên crop.
    Trả về: label(str), conf(float)
    """
    results = classifier.predict(
        source=crop_bgr,
        imgsz=CLS_IMGSZ,
        device=DEVICE,
        verbose=False
    )

    r0 = results[0]
    probs = r0.probs
    if probs is None:
        return "Unknown", 0.0

    top1 = int(probs.top1)
    conf = float(probs.top1conf)

    names = r0.names if hasattr(r0, "names") else {}
    label = names.get(top1, str(top1))
    return label, conf


def draw_overlay(frame, roi_rect, state, message, extra="", voting_progress=0.0):
    x1, y1, x2, y2 = roi_rect
    # Only draw ROI guide; remove on-screen status overlay
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


def draw_bbox_on_frame(frame, roi_offset_xy, xyxy_roi, text, color=(255, 0, 0)):
    if xyxy_roi is None:
        return
    ox, oy = roi_offset_xy
    x1, y1, x2, y2 = xyxy_roi
    gx1, gy1, gx2, gy2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy

    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, 2)
    cv2.putText(frame, text, (gx1, max(0, gy1 - 8)), FONT, 0.7, color, 2)


def majority_vote(labels):
    """Return (best_label, ratio)."""
    if not labels:
        return "Unknown", 0.0
    vals, counts = np.unique(np.array(labels, dtype=object), return_counts=True)
    best_i = int(np.argmax(counts))
    best = str(vals[best_i])
    ratio = float(counts[best_i]) / float(len(labels))
    return best, ratio


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # bạn đã sửa DSHOW
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam. Thử đổi CAMERA_INDEX (0/1/2...).")

    state = STATE_NO_OBJECT
    stable_start_time = None
    voting_start_time = None

    center_hist = deque(maxlen=5)
    vote_labels = deque(maxlen=VOTE_DEQUE_MAXLEN)

    # kết quả chốt
    final_xyxy = None
    final_det_conf = 0.0
    final_cls_label = "Unknown"
    final_cls_conf = 0.0

    print("Nhấn 'q' thoát | 'r' reset | 's' lưu ảnh ROI")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]
        roi_rect = compute_roi_rect(frame_w, frame_h)
        rx1, ry1, rx2, ry2 = roi_rect

        roi = frame[ry1:ry2, rx1:rx2].copy()

        has_obj, xyxy_roi, center_xy, det_conf, area_ratio, det_label = yolo_detect_in_roi(roi)
        now = time.time()

        # ============== STATE MACHINE ==============
        if state == STATE_NO_OBJECT:
            msg = "Dua rac vao khung do"
            extra = f"det_conf={det_conf:.2f} | lock>={CONF_LOCK:.2f} | area={area_ratio:.3f}"

            if has_obj:
                state = STATE_NOT_STABLE
                stable_start_time = now
                center_hist.clear()
                vote_labels.clear()
                if center_xy is not None:
                    center_hist.append(center_xy)

            draw_overlay(frame, roi_rect, state, msg, extra)

        elif state == STATE_NOT_STABLE:
            # Detect có nhưng conf chưa đủ lock => chưa cho vào voting
            if has_obj and det_conf < CONF_LOCK:
                msg = "Co vat nhung CHUA du tin cay (doi goc / dua sat hon)"
                extra = f"det_conf={det_conf:.2f} < lock {CONF_LOCK:.2f} | area={area_ratio:.3f}"

                stable_start_time = now
                center_hist.clear()
                vote_labels.clear()
                if center_xy is not None:
                    center_hist.append(center_xy)

                draw_overlay(frame, roi_rect, state, msg, extra)

            else:
                msg = "Giu yen vat trong khung (kiem tra on dinh)"
                extra = f"det_conf={det_conf:.2f} | area={area_ratio:.3f}"

                if not has_obj:
                    state = STATE_NO_OBJECT
                    stable_start_time = None
                    center_hist.clear()
                    vote_labels.clear()
                    draw_overlay(frame, roi_rect, state, "Dua rac vao khung do", extra)
                else:
                    if center_xy is not None:
                        center_hist.append(center_xy)

                    if len(center_hist) >= 2:
                        dx = abs(center_hist[-1][0] - center_hist[-2][0])
                        dy = abs(center_hist[-1][1] - center_hist[-2][1])
                        jitter = max(dx, dy)
                    else:
                        jitter = 999

                    if jitter <= MAX_CENTER_JITTER_PX:
                        if stable_start_time is None:
                            stable_start_time = now
                        if (now - stable_start_time) >= STABLE_SECONDS_REQUIRED:
                            state = STATE_VOTING
                            voting_start_time = now

                            # reset vote + final
                            vote_labels.clear()
                            final_xyxy = None
                            final_det_conf = 0.0
                            final_cls_label = "Unknown"
                            final_cls_conf = 0.0
                    else:
                        stable_start_time = now

                    extra += f" | jitter={jitter}px | need={STABLE_SECONDS_REQUIRED:.1f}s"
                    draw_overlay(frame, roi_rect, state, msg, extra)

        elif state == STATE_VOTING:
            # Nếu mất vật hoặc det_conf tụt dưới CONF_LOCK => quay lại NOT_STABLE
            if (not has_obj) or (det_conf < CONF_LOCK):
                state = STATE_NOT_STABLE
                voting_start_time = None
                stable_start_time = now
                center_hist.clear()
                vote_labels.clear()

                msg = "Voting bi huy (mat vat / conf thap). Thu lai!"
                extra = f"det_conf={det_conf:.2f} | need>={CONF_LOCK:.2f}"
                draw_overlay(frame, roi_rect, state, msg, extra)

            else:
                elapsed = now - (voting_start_time or now)
                progress = elapsed / VOTING_SECONDS

                # 1) crop bbox -> classifier
                crop = crop_with_padding(roi, xyxy_roi, pad_frac=CROP_PAD_FRAC)

                cls_label, cls_conf = "Unknown", 0.0
                if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                    cls_label, cls_conf = yolo_classify_crop(crop)

                # 2) Unknown threshold
                vote_label = cls_label if cls_conf >= CLS_UNKNOWN_CONF else "Unknown"
                vote_labels.append(vote_label)

                # 3) vote majority
                voted_label, voted_ratio = majority_vote(list(vote_labels))

                msg = "Dang nhan dien... (voting)"
                extra = (
                    f"det_conf={det_conf:.2f} | cls={cls_label}({cls_conf:.2f}) "
                    f"| vote={voted_label}({voted_ratio:.2f}) | {elapsed:.2f}/{VOTING_SECONDS:.2f}s"
                )

                # lưu kết quả tốt nhất trong voting (ưu tiên cls_conf)
                if cls_conf >= final_cls_conf:
                    final_xyxy = xyxy_roi
                    final_det_conf = det_conf
                    final_cls_label = cls_label
                    final_cls_conf = cls_conf

                # kết thúc voting -> chỉ vào RESULT nếu vote đủ mạnh + label không Unknown
                if elapsed >= VOTING_SECONDS:
                    if voted_label != "Unknown" and voted_ratio >= VOTE_MIN_RATIO:
                        state = STATE_RESULT
                        # chốt theo voted_label (ổn định hơn)
                        final_cls_label = voted_label
                    else:
                        state = STATE_NOT_STABLE
                        stable_start_time = now
                        vote_labels.clear()

                draw_overlay(frame, roi_rect, state, msg, extra, voting_progress=progress)

        elif state == STATE_RESULT:
            msg = "OK! Da nhan dien"
            extra = f"RESULT: {final_cls_label} | cls_conf~{final_cls_conf:.2f} | det_conf={final_det_conf:.2f} | 'r' reset"
            draw_overlay(frame, roi_rect, state, msg, extra)

        # ============== DRAW BBOX ==============
        if state != STATE_RESULT and has_obj and xyxy_roi is not None:
            # vẽ bbox realtime
            txt = f"det {det_conf:.2f}"
            draw_bbox_on_frame(frame, (rx1, ry1), xyxy_roi, txt, color=(255, 0, 0))

        if state == STATE_RESULT and final_xyxy is not None:
            # vẽ bbox + label kết quả
            txt = f"{final_cls_label} ({final_cls_conf:.2f})"
            draw_bbox_on_frame(frame, (rx1, ry1), final_xyxy, txt, color=(0, 255, 0))

        cv2.imshow("Garbage Classification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            state = STATE_NO_OBJECT
            stable_start_time = None
            voting_start_time = None
            center_hist.clear()
            vote_labels.clear()
            final_xyxy = None
            final_det_conf = 0.0
            final_cls_label = "Unknown"
            final_cls_conf = 0.0
        elif key == ord("s"):
            ts = int(time.time())
            roi_path = f"roi_capture_{ts}.jpg"
            cv2.imwrite(roi_path, roi)
            print("Da luu ROI:", roi_path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
