# %% 1. Read the tracking results and visual review
from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import pandas as pd

baseDir = Path(__file__).resolve().parent
resultDir = baseDir / "results"
outputDir = baseDir / "outputs"
runInfo = json.loads((resultDir / "run_info.json").read_text(encoding="utf-8"))
review = json.loads((baseDir / "data" / "manual_review.json").read_text(encoding="utf-8"))
manualDf = pd.read_csv(baseDir / "data" / "manual_people.csv")
trackDf = pd.read_csv(resultDir / "tracks.csv")
frameDf = pd.read_csv(resultDir / "frame_counts.csv")
if review["inputSha256"] != runInfo["inputSha256"]:
    raise ValueError("Visual review belongs to a different input video.")
if manualDf["personId"].duplicated().any() or len(manualDf) != review["manualCount"]:
    raise ValueError("Check the manual person ledger.")


# %% 2. Compare the counts
manualCount = len(manualDf)
trackerCount = int(trackDf["trackId"].nunique())
difference = trackerCount - manualCount
comparison = {"manualCount": manualCount, "uniqueTrackerIds": trackerCount,
              "difference": difference, "absoluteDifference": abs(difference),
              "countConvention": review["countConvention"],
              "note": "The visual count is a review estimate, not official dataset ground truth."}
(resultDir / "count_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
print("Manual visual count:", manualCount)
print("Unique tracker IDs:", trackerCount)
print("Tracker minus visual count:", difference)


# %% 3. Plot the count comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(["Visual count", "Tracker IDs"], [manualCount, trackerCount], color=["#1f77b4", "#ff7f0e"])
axes[0].set(title="Distinct people and distinct track IDs", ylabel="Count")
for index, value in enumerate([manualCount, trackerCount]):
    axes[0].text(index, value + 0.4, str(value), ha="center")
axes[0].set_ylim(0, max(manualCount, trackerCount) * 1.18)
axes[1].plot(frameDf["timeSeconds"], frameDf["uniqueTrackIds"], label="IDs seen so far", color="#ff7f0e")
axes[1].axhline(manualCount, color="#1f77b4", linestyle="--", label="Full-clip visual count")
axes[1].set(title="Cumulative tracker IDs", xlabel="Time (seconds)", ylabel="Count")
axes[1].legend()
fig.text(0.5, 0.02, "Re-entering people count once in visual review. Track IDs can restart; this is not an accuracy score.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(outputDir / "count_comparison.png", dpi=150)
plt.close(fig)


# %% 4. Check and show the observed ID change
switch = review["idSwitch"]
panels = []
cap = cv2.VideoCapture(str(baseDir / "data" / "vtest.avi"))
try:
    for frameNumber, trackId in [(switch["beforeFrame"], switch["oldId"]),
                                  (switch["afterFrame"], switch["newId"])]:
        selected = trackDf[(trackDf["frameNumber"] == frameNumber) & (trackDf["trackId"] == trackId)]
        if len(selected) != 1:
            raise ValueError("The reviewed ID change does not match this run. Review it again after changing the model/settings.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
        success, frame = cap.read()
        if not success:
            raise ValueError("Could not read an evidence frame.")
        row = selected.iloc[0]
        x1, y1, x2, y2 = [int(row[name]) for name in ["x1", "y1", "x2", "y2"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        label = f"ID {trackId}"
        cv2.putText(frame, label, (x1, max(20, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, label, (x1, max(20, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        frame = cv2.copyMakeBorder(frame, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(frame, f"Frame {frameNumber} | {frameNumber / runInfo['fps']:.1f}s | {switch['personId']}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        panels.append(frame)
finally:
    cap.release()
if not cv2.imwrite(str(outputDir / "id_switch.jpg"), cv2.hconcat(panels)):
    raise RuntimeError("Could not save the ID switch evidence.")
print("Observed ID change:", switch["oldId"], "to", switch["newId"], "at frame", switch["afterFrame"])
print(switch["observation"])
print("Likely reason:", switch["explanation"])
