import cv2
import ultralytics
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download
import imageio
ultralytics.checks()

def WallAngel(path):
    cap = cv2.VideoCapture("wallangel.mp4")
    assert cap.isOpened(), "Error reading video file"

    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("wallangel.output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Init AIGym
    gym = solutions.AIGym(
        show=True,  # Display the frame
        down_angle=80,
        up_angle=95,
        kpts=[6, 8, 10],  # keypoints index of person for monitoring specific exercise, by default it's for pushup. So for wall angel I used same keypoints, since we mostly track shoulder, elbow and arm position.
        model="yolo11n-pose.pt",  # Path to the YOLO11 pose estimation model file
        line_width=4,  # Adjust the line width for bounding boxes and text display
        verbose=False,
    )

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        results = gym(im0)  # monitor workouts on each frame
        video_writer.write(results.plot_im)  # write the output frame in file.

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

