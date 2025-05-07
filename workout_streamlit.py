from Lateral_Raise import lateral_raise
from Bench_Press import BenchPress
from shoulder_press import shoulder_press
from wall_angel import WallAngel
import streamlit as st
from upload import upload

st.title("Byte-Vision")
st.subheader("Let's track your workout form!") #im not creative :(
#replace with a better slogan later

st.sidebar.title("Workout Options") #temporary title for the sidebar
option = st.sidebar.selectbox(
    "Choose your exercise!",
    [" ","Lateral Raise", "Bench Press", "Shoulder Press", "Wall Angel"]
)
if option == " ": #default
    st.write("Please select an exercise from the sidebar.")
    st.stop()

st.write(f"You selected {option}. A webcam window will open - press 'q' to quit. May need to press 'q' multiple times.")

if option == "Lateral Raise":
    st.write("You selected Lateral Raise")
    lateral_raise()

elif option == "Bench Press":
    st.write("You selected Bench Press")
    BenchPress()

elif option == "Shoulder Press":
    st.write("You selected Shoulder Press")
    vid_path = upload()
    if vid_path and st.button("Start Shoulder Press Analysis"):
        shoulder_press(vid_path)
    
elif option == "Wall Angel":
    st.write("You selected Wall Angel")
    WallAngel()

st.stop()    