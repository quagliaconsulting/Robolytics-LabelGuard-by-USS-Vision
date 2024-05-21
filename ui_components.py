# ui_components.py
import streamlit as st
import base64

def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def add_logos():
    ultralytics_logo_base64 = get_image_base64('./logos/ultralytics.png')
    uss_logo_base64 = get_image_base64('./logos/uss.png')
    roboflow_logo_base64 = get_image_base64('./logos/roboflow.png')

    st.markdown(
        f"""
        <style>
        .logo-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 50px;  /* Increase spacing between images */
        }}
        .logo-container img {{
            max-height: 100%;
            max-width: 100%;
        }}
        .ultralytics-logo {{
            width: 200px;  /* Adjust width as needed */
        }}
        .uss-logo {{
            width: 200px;  /* Adjust width as needed */
        }}
        .roboflow-logo {{
            width: 200px;  /* Adjust width as needed */
        }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{ultralytics_logo_base64}" class="ultralytics-logo" alt="Ultralytics">
            <img src="data:image/png;base64,{uss_logo_base64}" class="uss-logo" alt="USS Vision">
            <img src="data:image/png;base64,{roboflow_logo_base64}" class="roboflow-logo" alt="Roboflow">
        </div>
        """,
        unsafe_allow_html=True
    )

def main_ui():
    st.subheader("Robolytics LabelGuard by USS Vision", divider='rainbow')
    st.write("Identify likely mislabeled objects in your object detection dataset.")
