from pathlib import Path
import math
import json, shutil, uuid, time
from typing import Optional
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
RESULTS = ROOT / "results"
UPLOADS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

app = FastAPI(title="TrafficVision AI Prototype")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# COCO class IDs used by common YOLO models
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRAVEL_DIRECTIONS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}

def load_model():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception:
        return None

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/status")
def status():
    return {"ok": True, "model": "YOLOv8n" if load_model() else "unavailable",
            "note": "First model initialization may download weights."}

@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), sample_seconds: float = Form(1.0),
                  travel_direction: Optional[str] = Form(None)):
    if not video.filename or Path(video.filename).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, "Upload a supported video file (MP4, MOV, AVI, MKV, WEBM).")
    if not math.isfinite(sample_seconds) or not 0.5 <= sample_seconds <= 10:
        raise HTTPException(400, "Sampling interval must be between 0.5 and 10 seconds.")
    if travel_direction and travel_direction not in TRAVEL_DIRECTIONS:
        raise HTTPException(400, "Select a valid travel direction.")
    job = uuid.uuid4().hex[:10]
    path = UPLOADS / f"{job}_{Path(video.filename).name}"
    with path.open("wb") as out:
        shutil.copyfileobj(video.file, out)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read this video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0
    interval = max(1, int(fps * sample_seconds))
    model = load_model()
    detections, previews, total = [], [], 0
    violations, track_starts, flagged_tracks = [], {}, set()
    direction = TRAVEL_DIRECTIONS.get(travel_direction)
    tracking_interval = max(1, round(fps / 5))
    tracking_failed = False
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        evidence_due = frame_no % interval == 0
        tracking_due = direction and frame_no % tracking_interval == 0
        if evidence_due or tracking_due:
            stamp = frame_no / fps if fps else 0
            found, frame_violations = [], []
            if model:
                try:
                    options = dict(verbose=False, conf=0.35, imgsz=640)
                    if direction and not tracking_failed:
                        result = model.track(frame, persist=True, tracker="bytetrack.yaml", **options)[0]
                    else:
                        result = model.predict(frame, **options)[0]
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls in VEHICLE_CLASSES:
                            xyxy = [round(float(x), 1) for x in box.xyxy[0].tolist()]
                            found.append({"type": VEHICLE_CLASSES[cls], "confidence": round(float(box.conf[0]), 3), "box": xyxy})
                            if direction and box.id is not None:
                                track_id = int(box.id[0])
                                found[-1]["track_id"] = track_id
                                height, width = frame.shape[:2]
                                position = ((xyxy[0] + xyxy[2]) / (2 * width), xyxy[3] / height)
                                start_time, start = track_starts.setdefault(track_id, (stamp, position))
                                dx, dy = position[0] - start[0], position[1] - start[1]
                                along = dx * direction[0] + dy * direction[1]
                                across = dx * direction[1] - dy * direction[0]
                                if (track_id not in flagged_tracks and stamp - start_time >= 0.8
                                        and -along >= 0.10 and -along >= 1.5 * abs(across)):
                                    flagged_tracks.add(track_id)
                                    frame_violations.append({
                                        "rule": "wrong_way_travel", "track_id": track_id,
                                        "vehicle": VEHICLE_CLASSES[cls], "start_time": round(start_time, 2),
                                        "time": round(stamp, 2),
                                    })
                except Exception:
                    tracking_failed = bool(direction)
            if not evidence_due and not frame_violations:
                frame_no += 1
                continue
            total += len(found)
            # Save a small annotated frame for reviewer evidence
            for item in found:
                x1,y1,x2,y2 = map(int, item["box"])
                flagged = item.get("track_id") in flagged_tracks
                color = (40, 75, 235) if flagged else (30, 190, 80)
                label = f'{item["type"]} {item["confidence"]:.2f}'
                if flagged:
                    label += f' #{item["track_id"]} wrong way?'
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1,max(20,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            image_name = f"{job}_{len(previews)}.jpg"
            cv2.imwrite(str(RESULTS / image_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            for violation in frame_violations:
                violation["image"] = f"/api/frame/{image_name}"
                violations.append(violation)
            previews.append({"time": round(stamp, 2), "image": f"/api/frame/{image_name}", "vehicles": found})
        frame_no += 1
    cap.release()
    report = {
        "job_id": job, "filename": video.filename, "duration_seconds": round(duration, 2),
        "sample_interval_seconds": sample_seconds, "sampled_frames": len(previews),
        "vehicle_detections": total, "model_available": model is not None,
        "frames": previews,
        "travel_direction": travel_direction or None,
        "wrong_way_check": ("disabled" if not direction else
                            "unavailable" if not model else
                            "incomplete" if tracking_failed else "completed"),
        "violations": violations,
        "review_notice": "Automated detections are suggestions only. Review every frame; this prototype does not establish a traffic offence or issue an official challan."
    }
    (RESULTS / f"{job}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

@app.get("/api/frame/{name}")
def get_frame(name: str):
    safe = Path(name).name
    file = RESULTS / safe
    if not file.exists(): raise HTTPException(404, "Frame not found")
    return FileResponse(file, media_type="image/jpeg")

@app.get("/api/report/{job}")
def get_report(job: str):
    file = RESULTS / f"{Path(job).name}.json"
    if not file.exists(): raise HTTPException(404, "Report not found")
    return FileResponse(file, media_type="application/json", filename=f"trafficvision-{job}.json")
