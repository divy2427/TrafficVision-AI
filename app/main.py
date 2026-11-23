from pathlib import Path
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
async def analyze(video: UploadFile = File(...), sample_seconds: float = Form(1.0)):
    if not video.filename or Path(video.filename).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, "Upload a supported video file (MP4, MOV, AVI, MKV, WEBM).")
    if sample_seconds < 0.5 or sample_seconds > 10:
        raise HTTPException(400, "Sampling interval must be between 0.5 and 10 seconds.")
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
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_no % interval == 0:
            stamp = frame_no / fps if fps else 0
            found = []
            if model:
                try:
                    result = model.predict(frame, verbose=False, conf=0.35, imgsz=640)[0]
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls in VEHICLE_CLASSES:
                            xyxy = [round(float(x), 1) for x in box.xyxy[0].tolist()]
                            found.append({"type": VEHICLE_CLASSES[cls], "confidence": round(float(box.conf[0]), 3), "box": xyxy})
                except Exception:
                    pass
            total += len(found)
            # Save a small annotated frame for reviewer evidence
            for item in found:
                x1,y1,x2,y2 = map(int, item["box"])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (30,190,80), 2)
                cv2.putText(frame, f'{item["type"]} {item["confidence"]:.2f}', (x1,max(20,y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (30,190,80), 2)
            image_name = f"{job}_{len(previews)}.jpg"
            cv2.imwrite(str(RESULTS / image_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            previews.append({"time": round(stamp, 2), "image": f"/api/frame/{image_name}", "vehicles": found})
        frame_no += 1
    cap.release()
    report = {
        "job_id": job, "filename": video.filename, "duration_seconds": round(duration, 2),
        "sample_interval_seconds": sample_seconds, "sampled_frames": len(previews),
        "vehicle_detections": total, "model_available": model is not None,
        "frames": previews,
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
