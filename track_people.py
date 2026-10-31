# %% 1. Import the libraries
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import subprocess

import cv2
import imageio_ffmpeg
import pandas as pd
from ultralytics import YOLO


# %% 2. Set the paths
baseDir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Track people using YOLO11n and ByteTrack.")
parser.add_argument("--input", type=Path, default=baseDir / "data" / "vtest.avi")
args = parser.parse_args()
inputPath = args.input.resolve()
outputDir = baseDir / "outputs"
resultDir = baseDir / "results"
modelDir = baseDir / "models"
for folder in [outputDir, resultDir, modelDir]:
    folder.mkdir(exist_ok=True)

rawPath = outputDir / "people_tracked_raw.mp4"
outputPath = outputDir / "people_tracked.mp4"
trackerPath = baseDir / "bytetrack.yaml"
confidenceThreshold = 0.1
if not inputPath.is_file():
    raise FileNotFoundError(f"Video not found: {inputPath}")
if inputPath in [rawPath.resolve(), outputPath.resolve()]:
    raise ValueError("Choose an input outside the output video paths.")


# %% 3. Load the model and read the video
modelPath = modelDir / "yolo11n.pt"
model = YOLO(str(modelPath))
cap = cv2.VideoCapture(str(inputPath))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not cap.isOpened() or width <= 0 or height <= 0 or not 0 < fps < 1000:
    cap.release()
    raise ValueError("Could not read valid video details.")

writer = cv2.VideoWriter(str(rawPath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not writer.isOpened():
    cap.release()
    writer.release()
    raise RuntimeError("Could not create the output video.")
print("Input:", width, "x", height, "| FPS:", fps, "| Frames:", totalFrames)


# %% 4. Track people and store each ID once
seenIds = set()
trackRows = []
frameRows = []
frameNumber = 0
try:
    while True:
        success, frame = cap.read()
        if not success:
            break
        result = model.track(frame, persist=True, classes=[0], conf=confidenceThreshold,
                             tracker=str(trackerPath), imgsz=640, iou=0.7,
                             device="cpu", verbose=False)[0]
        peopleInFrame = 0
        if result.boxes.is_track:
            for box in result.boxes:
                trackId = int(box.id.item())
                score = float(box.conf.item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                seenIds.add(trackId)
                peopleInFrame += 1
                trackRows.append({"frameNumber": frameNumber, "timeSeconds": frameNumber / fps,
                                  "trackId": trackId, "confidence": score,
                                  "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                color = (40 + trackId * 43 % 190, 70 + trackId * 61 % 170, 50 + trackId * 89 % 190)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID {trackId}"
                labelX = max(0, min(int(x1), width - 75))
                labelY = min(height - 8, max(58, int(y1) - 8))
                cv2.putText(frame, label, (labelX, labelY), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
                cv2.putText(frame, label, (labelX, labelY), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        frameRows.append({"frameNumber": frameNumber, "timeSeconds": frameNumber / fps,
                          "peopleInFrame": peopleInFrame, "uniqueTrackIds": len(seenIds)})
        cv2.rectangle(frame, (0, 0), (width, 44), (25, 25, 25), -1)
        cv2.putText(frame, f"Frame {frameNumber} | Visible tracks: {peopleInFrame} | Unique IDs: {len(seenIds)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
        writer.write(frame)
        if frameNumber in [0, totalFrames // 3, totalFrames - 1]:
            if not cv2.imwrite(str(outputDir / f"tracked_frame_{frameNumber:04d}.jpg"), frame):
                raise RuntimeError("Could not save a preview.")
        frameNumber += 1
        if frameNumber % 100 == 0:
            print("Processed frames:", frameNumber)
finally:
    cap.release()
    writer.release()
if frameNumber != totalFrames or frameNumber == 0:
    raise ValueError("The decoded frame count does not match the input.")


# %% 5. Save the tracking tables
trackDf = pd.DataFrame(trackRows, columns=["frameNumber", "timeSeconds", "trackId", "confidence", "x1", "y1", "x2", "y2"])
frameDf = pd.DataFrame(frameRows)
trackDf.to_csv(resultDir / "tracks.csv", index=False)
frameDf.to_csv(resultDir / "frame_counts.csv", index=False)
trackSummary = trackDf.groupby("trackId").agg(firstFrame=("frameNumber", "min"),
                                            lastFrame=("frameNumber", "max"),
                                            observedFrames=("frameNumber", "size"))
trackSummary.to_csv(resultDir / "track_summary.csv")
runInfo = {"model": "yolo11n.pt", "tracker": "ByteTrack", "confidence": confidenceThreshold,
           "imageSize": 640, "nmsIou": 0.7, "classes": [0], "device": "cpu",
           "width": width, "height": height, "fps": fps, "frames": frameNumber,
           "uniqueTrackIds": len(seenIds), "trackIds": sorted(seenIds), "frameIndexing": "zero-based",
           "inputSha256": hashlib.sha256(inputPath.read_bytes()).hexdigest(),
           "modelSha256": hashlib.sha256(modelPath.read_bytes()).hexdigest(),
           "trackerSha256": hashlib.sha256(trackerPath.read_bytes()).hexdigest(),
           "packages": {name: importlib.metadata.version(name) for name in
                        ["ultralytics", "torch", "torchvision", "opencv-python", "numpy", "lap"]}}
(resultDir / "run_info.json").write_text(json.dumps(runInfo, indent=2) + "\n", encoding="utf-8")


# %% 6. Save a browser-compatible output video
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(rawPath), "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(outputPath)], check=True)
print("Unique tracker IDs:", len(seenIds))
print("These are track IDs, not guaranteed unique real people.")
print("Saved:", outputPath)
