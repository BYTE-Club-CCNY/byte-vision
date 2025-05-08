import cv2
import ultralytics
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download
import imageio
ultralytics.checks()

def WallAngel():
    video_writer = None
    cap = None
    try:
        cap = cv2.VideoCapture("vids/WallAngelDemo.mp4") #fixing the path to the video file
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("wallangel.output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init AIGym
        gym = solutions.AIGym(
            #show=True,  # Display the frame
            down_angle=115,
            up_angle=140,
            kpts=[6, 8, 10],  # keypoints index of person for monitoring specific exercise, by default it's for pushup. So for wall angel I used same keypoints, since we mostly track shoulder, elbow and arm position.
            model="yolo11n-pose.pt",  # Path to the YOLO11 pose estimation model file
            line_width=4,  # Adjust the line width for bounding boxes and text display
            verbose=False,
        )

        # Process video and implement logic
        min_angle = 95
        max_angle = 130
        angle_range_up = False
        angle_range_down = False
        prev_state = None
        warning_message = ""
        step = 0

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = gym(im0)  # monitor workouts on each frame
            video_writer.write(results.plot_im)  # write the output frame in file.
            cv2.imshow("Wall Angel Tracker", results.plot_im)

            angle = results.workout_angle[0]
            state = results.workout_stage[0]

            if state != prev_state and step!=0:
                if (angle_range_up==False) and state=="down":
                    warning_message = "You didn't raise your hands high enough last time."
                    bad_form = True
                    print(warning_message, state)

                if (angle_range_down==False) and state=="up":
                    warning_message = "You didn't lower your hands down enough last time."
                    bad_form = True
                    print(warning_message, state)

                angle_range_up, angle_range_down = False, False
                
            if state == "up":
                if angle >= max_angle:
                    angle_range_up = True
            elif state == "down":
                if angle <= min_angle:
                    angle_range_down = True

            prev_state = state


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    WallAngel()
WallAngel()