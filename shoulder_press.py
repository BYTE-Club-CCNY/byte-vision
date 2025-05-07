from ultralytics import solutions
import cv2
import streamlit as st
from collections import deque
import subprocess
import os

def shoulder_press(vid_path):
    video_writer = None
    cap = None
    try:
        cap = cv2.VideoCapture(vid_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        os.makedirs("output_vids", exist_ok=True)
        cwd = os.getcwd()
        output_avi = os.path.join(cwd, "output_vids/Shoulderpress.demo.video.output.avi")
        output_mp4 = os.path.join(cwd, "output_vids/Shoulderpress.demo.video.output.mp4")

        if os.path.exists(output_avi):
            os.remove(output_avi)

        if os.path.exists(output_mp4):
            os.remove(output_mp4)

        video_writer = cv2.VideoWriter(output_avi, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        gym = solutions.AIGym(
            show=True,
            kpts=[5, 7, 9],
            model="yolo11n-pose.pt",  # Path to the YOLO11 pose estimation model file
            line_width=4,  # Adjust the line width for bounding boxes and text display
            up_angle=127,
            down_angle=95,
            verbose=False,
        )

        queue = deque(maxlen=2*fps) # contains the last state
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                st.toast('Video frame is empty or video processing has been successfully completed.')
                print('Video frame is empty or video processing has been successfully completed.')
                break
            results = gym(im0)
            curr_stage = results.workout_stage[0]
            queue.append(curr_stage)
            if len(queue) == queue.maxlen and len(set(queue)) == 1:
                st.toast("Check your form.")
                print('Check your form.')
            video_writer.write(results.plot_im)

            if st.button('Stop'):
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        video_writer.release()

        if os.path.exists(vid_path):
            os.remove(vid_path)

        if os.path.exists(output_mp4):
            os.remove(output_mp4)

        subprocess.run([
            'ffmpeg', '-i', output_avi,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-y',
            output_mp4
        ])

if __name__ == "__main__":
    shoulder_press()