# Week 4: Tracking People

## Detection and tracking

Detection finds people separately in each frame. Tracking tries to keep the same ID for a person across frames.

An ID is the tracker's label. It is not a person's name, and it is not guaranteed to stay correct.

## Keeping IDs between frames

`model.track(frame, persist=True, classes=[0], tracker=str(trackerPath))`

`persist=True` tells the tracker that the next frame follows the previous one. `classes=[0]` keeps the person class.

This project passes its included `bytetrack.yaml` explicitly. It does not depend on whichever tracker a library version selects by default.

Read an assigned ID with `box.id`. Check `result.boxes.is_track` before reading IDs because a frame can have no tracked boxes.

## ByteTrack

ByteTrack first matches higher-confidence detections with existing tracks using motion and box overlap. It then tries lower-confidence detections for tracks left unmatched.

It does not accept every possible low score. The detector cutoff and the tracker's confidence thresholds decide which boxes reach each stage. Lower-confidence boxes can help recover an existing track without starting a new one.

## Counting once

Add every observed ID to `seenIds`, a Python set. `len(seenIds)` is the number of different tracker IDs seen so far.

This is an estimate of unique people. Adding the people in every frame would count the same person repeatedly.

## ID switches and broken tracks

An ID switch assigns the wrong identity to a person, such as swapping two people's IDs. The class also uses the term when the same person receives a new ID.

A track can break during an occlusion or missed detection. When that person is detected again, a new ID may start. This is track fragmentation. A person leaving and later returning can also receive another ID.

These events can count one person more than once. A false person detection can add an extra ID. Missed people or two people being treated as one track can lower the count.

## Comparing with a manual count

Review the video and follow visible people using their appearance and movement. Count each distinct person once and record any uncertain identity matches.

A visual count is a reference, not perfect ground truth: distant people, occlusion and re-entry can make it uncertain. Explain the observed difference instead of assuming every tracker ID represents a different person.

## Reference

[Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) explains the tracking arguments, ID access and confidence thresholds. The included configuration records the settings used for this project.
