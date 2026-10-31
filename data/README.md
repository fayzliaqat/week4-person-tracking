# Sample video

`vtest.avi` is the unchanged sample video distributed in the official OpenCV repository. It shows pedestrians from a fixed camera and is clearer for visual counting than the defocused sample used in the previous assignments.

- [Source file in OpenCV](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi)
- [Direct source](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi)
- [OpenCV licence and notices](https://github.com/opencv/opencv/blob/4.x/LICENSE), also preserved in `OPENCV_LICENSE.txt`

The original source is included without cropping, resizing, adding people, or changing its frame rate. Its SHA-256 is recorded in `results/run_info.json`. The tracking output is generated separately. It has 795 frames, 768 x 576 resolution, and 10 FPS (79.5 seconds).

`manual_people.csv` records visual person identities, independently of tracker IDs. `manual_review.json` describes the count convention and the observed ID switch. These are visual review notes, not official dataset ground truth or a complete MOT benchmark annotation.
