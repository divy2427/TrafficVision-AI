# TrafficVision AI — working local web prototype

A browser-based educational prototype that accepts a short traffic video, samples frames, runs YOLO vehicle detection, displays timestamped evidence frames, and exports a JSON report.

## What works
- Browser video upload (MP4, MOV, AVI, MKV, WEBM; prototype cap 300 MB)
- Configurable frame sampling interval
- YOLOv8n detection for cars, motorcycles, buses, and trucks
- Timestamped annotated evidence frames
- Downloadable JSON analysis report
- Possible wrong-way travel with timestamped evidence. Use a fixed-camera clip of a single direction of traffic and select the allowed on-screen direction before analyzing.

## What is not implemented
This is an MVP, not a production enforcement system. It does **not** perform reliable number-plate localization/OCR, determine a legal violation, identify a registered owner, calculate official fines, or issue a challan. Those require jurisdiction-specific rules, validated models, secure integrations, and human/legal review. Do not treat model output as proof.

## Requirements
- Python 3.10 or 3.11 recommended
- 8 GB RAM recommended (16 GB is more comfortable)
- Internet for the first installation and initial YOLO model-weight download
- Windows, macOS, or Linux

## Run on Windows
1. Install Python 3.10/3.11 and ensure Python is on PATH.
2. Extract this ZIP.
3. Double-click `run_windows.bat`.
4. Wait for packages and the YOLO model weights to install/download.
5. Open http://127.0.0.1:8000 in your browser.

## Run on macOS / Linux
1. Install Python 3.10/3.11.
2. Extract the ZIP and open a terminal in the extracted folder.
3. Run `chmod +x run_mac_linux.sh && ./run_mac_linux.sh`.
4. Open http://127.0.0.1:8000 in your browser.

The first run can take several minutes. If YOLO cannot load, the web app still starts but will show zero detections and mark the detector unavailable. Check the terminal for the installation/model download error.

## Privacy and safety
- This prototype binds only to `127.0.0.1` (your own computer); do not expose it publicly.
- Use only footage you have permission to process.
- Avoid uploading footage containing children or unrelated bystanders where possible.
- The app stores uploaded videos in `uploads/` and sampled frames/reports in `results/`. Delete these folders when finished.
- Review all outputs. A false positive is possible.

## Important model licensing note
The included Ultralytics package and YOLO model are subject to Ultralytics' licensing terms (commonly AGPL-3.0 for the open-source offering, with separate commercial licensing options). Review the current license before redistribution, private/closed-source deployment, or commercial use. This ZIP is intended for learning and a local prototype.

## Project layout
```
trafficvision_ai_prototype/
  app/
    main.py
    static/index.html
  uploads/       created automatically
  results/       created automatically
  requirements.txt
  run_windows.bat
  run_mac_linux.sh
  README.md
```

## Next development milestones
1. Add plate-region detection and OCR, with confidence scores and manual correction.
2. Add a reviewer workflow for classifying potential violations.
3. Store reviewed cases in a database with role-based access and audit logs.
4. Add a configurable fine reference table only after verifying current local law.
5. Add PDF evidence reports and secure retention/deletion controls.
