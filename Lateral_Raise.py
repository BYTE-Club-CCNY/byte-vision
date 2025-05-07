import cv2
import ultralytics
from ultralytics import YOLO
from ultralytics import solutions
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.downloads import safe_download



def lateral_raise():
    video_writer = None
    cap = None
    try:
        model = YOLO("yolo11n-pose.pt")

        #cap = cv2.VideoCapture("Pushups.demo.video.mp4")
        cap = cv2.VideoCapture("vids/LatRaiseDemo.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("LatRaise.demo.video.output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


        def new_process(self, im0):
                annotator = SolutionAnnotator(im0, line_width=self.line_width)  

                self.extract_tracks(im0)  
                tracks = self.tracks[0]

                if tracks.boxes.id is not None:
                    if len(tracks) > len(self.count): 
                        new_human = len(tracks) - len(self.count)
                        self.angle += [0] * new_human
                        self.count += [0] * new_human
                        self.stage += ["-"] * new_human

                    
                    for ind, k in enumerate(reversed(tracks.keypoints.data)):
                    
                        kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                        self.angle[ind] = annotator.estimate_pose_angle(*kpts)
                        annotator.draw_specific_kpts(k, self.kpts, radius=self.line_width * 3)

                        ##THIS PART WAS CHANGED, SIGNS WERE FLIP-FLOPPED.
                        if self.angle[ind] > self.down_angle:
                            if self.stage[ind] == "up":
                                self.count[ind] += 1
                            self.stage[ind] = "down"
                        elif self.angle[ind] < self.up_angle:
                            self.stage[ind] = "up"

                        
                        if self.show_labels:
                            annotator.plot_angle_and_count_and_stage(
                                angle_text=self.angle[ind],  
                                count_text=self.count[ind],  
                                stage_text=self.stage[ind],  
                                center_kpt=k[int(self.kpts[1])],  
                            )
                plot_im = annotator.result()
                self.display_output(plot_im)  

                # Return SolutionResults
                return SolutionResults(
                    plot_im=plot_im,
                    workout_count=self.count,
                    workout_stage=self.stage,
                    workout_angle=self.angle,
                    total_tracks=len(self.track_ids),
                )

        #solutions.AIGym.process = new_process

        othergym = solutions.AIGym(
            #show=True,  # Display the frame
            kpts=[7, 0, 8],  # keypoints index of person for monitoring specific exercise, by default it's for pushup
            model="yolo11n-pose.pt",  # Path to the YOLO11 pose estimation model file
            line_width=4,  # Adjust the line width for bounding boxes and text display
            verbose=False,
            up_angle = 120,
            down_angle = 70         
        )

        #print('dog')


        # Process video

        count = 0
        partial_detector = False
        #0 for at bottom
        #1 for in progress
        #-999 for waiting for user to get into angle.

        while cap.isOpened():

            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            results = othergym(im0)  # monitor workouts on each frame

            video_writer.write(results.plot_im)  # write the output frame in file.

            if(partial_detector == True and results.workout_angle[0] > 110):
                print("Did a partial!")
            
            if(results.workout_stage[0] == 'up' and results.workout_angle[0] < 100): 
                partial_detector = True

            if(results.workout_stage[0] == 'down'):
                partial_detector = False
            
            if(results.workout_angle[0] > 80):
                count = count + 1
            else:
                count = 0

            if(count > 200):
                print("Please fix your form!")

            
            

            print(partial_detector)

            #when hit up, flip hit top
            #if angle up angle is passed again when bool is flipped, partial detected
            #reset bool at bottom.
        
            cv2.imshow("Lateral Raise Tracker", results.plot_im)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break    

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        video_writer.release()


if __name__ == "__main__":
    lateral_raise() 