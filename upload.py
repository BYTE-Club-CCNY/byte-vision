import streamlit as st
import os

def upload():
    try:
        vid = st.file_uploader('Uploads a workout video', type=['mp4', 'mov'])

        if vid is None:
            st.write("Please upload a video file.")
            return
        
        cwd = os.getcwd()
        vid_path = os.path.join(cwd, f'uploads/{vid.name}')

        with open(vid_path, "wb") as f:
            f.write(vid.read())
        
        return vid_path
    except Exception as e:
        st.error(f'Error: {e}')