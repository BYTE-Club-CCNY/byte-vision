def BenchPress():
    video_writer = None
    cap = None
    try:
        import cv2, ultralytics, os, subprocess
        from ultralytics import solutions
        from collections import deque

        video_path = "vids/BenchpressDemo.mp4"
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
        ))

        cwd = os.getcwd()
        output_avi = os.path.join(cwd, "BenchpressDemoOutput.avi")
        output_mp4 = os.path.join(cwd, "BenchpressDemoOutput.mp4")

        video_writer = cv2.VideoWriter(
            output_avi,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        # Init AIGym
        gym = solutions.AIGym(
            show=False,
            kpts=[6, 8, 10],
            down_angle=20,
            up_angle=130,
            model="yolo11n-pose.pt",
            line_width=4,
            verbose=False,
        )

        # --- Bad rep tracking state ---
        min_angle, max_angle = 180, 0
        bad_rep_warned, show_warning_text = False, False
        last_state = None
        up_frames, down_frames = 0, 0
        warning_message = ""

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = gym(im0)
            angle = results.workout_angle[0]
            state = results.workout_stage[0]

            # --- Bad rep detection logic ---
            if state == "up":
                if last_state != "up":
                    up_frames = 0  

                if last_state == "down":
                    min_angle = 180
                    bad_rep_warned, show_warning_text = False, False

                up_frames += 1
                time_in_up = up_frames / fps
                min_angle = min(min_angle, angle)

                if time_in_up > 5 and angle > min_angle + 40 and min_angle > gym.down_angle and not bad_rep_warned:
                    warning_message = "Not going low enough, please check your form."
                    show_warning_text, bad_rep_warned = True, True

            elif state == "down":
                if last_state != "down":
                    down_frames = 0

                if last_state == "up":
                    max_angle = 0
                    bad_rep_warned, show_warning_text = False, False

                down_frames += 1
                time_in_down = down_frames / fps
                max_angle = max(max_angle, angle)

                if time_in_down > 5 and angle < max_angle - 30 and max_angle < gym.up_angle and not bad_rep_warned:
                    warning_message = "Not going high enough, please check your form"
                    show_warning_text, bad_rep_warned = True, True

            last_state = state

            if show_warning_text:
                cv2.putText(results.plot_im, warning_message, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("Bench Press Tracker", results.plot_im)
            video_writer.write(results.plot_im)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if cap:
            cap.release()
        if video_writer:
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