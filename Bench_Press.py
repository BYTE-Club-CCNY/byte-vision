def BenchPress():
    video_writer = None
    cap = None
    try:
        import cv2
        import ultralytics
        from ultralytics import solutions
        from collections import deque
        import os
        import subprocess

        video_path = "vids/Benchpress.demo.video.mp4"
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
        ))

        cwd = os.getcwd()
        output_avi = os.path.join(cwd, "Benchpress.demo.video.output.avi")
        output_mp4 = os.path.join(cwd, "Benchpress.demo.video.output.mp4")
        
        video_writer = cv2.VideoWriter(
            output_avi,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        # Init AIGym
        gym = solutions.AIGym(
            #show=True,                  # Display the frame
            kpts=[6, 8, 10],            # keypoints index of person for monitoring specific exercise, by default it's for pushup
            down_angle=20,
            model="yolo11n-pose.pt",    # Path to the YOLO11 pose estimation model file
            line_width=4,               # Adjust the line width for bounding boxes and text display
            verbose=False,
        )

        queue = deque(maxlen=fps)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            
            results = gym(im0)
            curr_stage = results.workout_stage[0]
            queue.append(curr_stage)

            if len(queue) == queue.maxlen and len(set(queue)) == 1:
                print("Check your form.")

            video_writer.write(results.plot_im)
            cv2.imshow("Bench Press Tracker", results.plot_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        # Convert .avi to .mp4 using ffmpeg
        subprocess.run([
            'ffmpeg', '-i', output_avi,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_mp4
        ])

if __name__ == "__main__":
    BenchPress()