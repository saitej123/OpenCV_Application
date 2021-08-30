import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 270px;
        margin-left: -270px;
    }
    </style>
    """,
    unsafe_allow_html=True,)

st.markdown("Application Developed by Sai Tej ! 🚀")

st.subheader("""OpenCV  methods""")



uploaded_file = st.file_uploader("Choose a image file", type=['jpg', 'jpeg', 'png', 'jfif','PNG','JPEG','JPG'])
                                
# Useful methods in opencv
st.sidebar.header("Play with the methods on your image")


# rotate an image by an angle
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv2.warpAffine(img, rotMat, dimensions)


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    c1, c2 = st.columns(2)
    with c1: # Now do something with the image! For example, let's display it:
        st.write("""### Original image:""")
        st.image(opencv_image, channels="BGR")
        

    # Color conversions
    st.sidebar.subheader("Color conversions")
    selected_color_conv = st.sidebar.selectbox("Select", ["GRAY", "HSV"])
    with c2:
        if selected_color_conv == "GRAY":
            colored_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            st.write("""### Grayscale image:""")            
            st.image(colored_img)
        elif selected_color_conv == "HSV":
            colored_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
            st.write("""### Hue Saturated image:""")
            
            st.image(colored_img)
    st.code("""
            colored_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
            """, language='python')
    # Gaussian blur
    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    st.sidebar.subheader("Gaussian Blur")
    selected_kernel_size = st.sidebar.slider('Kernel size', 1, 49, step=2)
    kernel_size = (selected_kernel_size, selected_kernel_size)
    blurred_img = cv2.GaussianBlur(
        opencv_image_rgb, kernel_size, cv2.BORDER_DEFAULT)
    st.write("""### Gaussian Blur image:""")
    st.write(f"Kernel size: {(kernel_size)}")
    st.code("""
    blurred_img = cv2.GaussianBlur(opencv_image_rgb, kernel_size, cv2.BORDER_DEFAULT)
    """, language='python')
    st.image(blurred_img)
    
    # Adaptive threshold (default Binary)
    st.sidebar.subheader("Adaptive Thresholding ")
    threshold_min = st.sidebar.slider("Threshold_min", 0, 255, step=1, value=100)
    threshold_max = st.sidebar.slider("Threshold_max", 0, 255, step=1, value=255)
    selected_image_type = st.sidebar.selectbox(
        "Apply on:", ["Original image", "Blurred image"])
    
    if selected_image_type.startswith("Original"):
        _,thrsh_img = cv2.threshold(opencv_image_rgb, threshold_min, threshold_max, cv2.THRESH_BINARY)
    else:
        _, thrsh_img = cv2.threshold(
             blurred_img, threshold_min, threshold_max, cv2.THRESH_BINARY)
    st.write("""### Adaptive threshold :""")
    st.code("""
    thrsh_img = cv2.threshold(opencv_image_rgb, threshold_min, threshold_max, cv2.THRESH_BINARY)
    """, language='python')
    st.image(thrsh_img)
    
    
    

    # Canny edges
    st.sidebar.subheader("Canny edges")
    threshold1 = st.sidebar.slider("Threshold 1", 1, 500, step=1, value=300)
    threshold2 = st.sidebar.slider("Threshold 2", 1, 500, step=1, value=200)
    selected_canny_img = st.sidebar.selectbox(
        "Apply on:", ["Original image", "Blurred image","Threshold image"])
    if selected_canny_img.startswith("Original"):
        canny_img = cv2.Canny(opencv_image_rgb, threshold1, threshold2)
    elif selected_canny_img.startswith("Threshold"):
        canny_img = cv2.Canny(thrsh_img, threshold1, threshold2)
    else:
        canny_img = cv2.Canny(blurred_img, threshold1, threshold2)
    st.write("""### Canny edges detector:""")
    st.code("""
    canny_img = cv2.Canny(opencv_image_rgb, threshold1, threshold2)
    """, language='python')
    st.image(canny_img)
    
       

    # dilated image
    st.sidebar.subheader("Dilated image")
    dilated_kernel = st.sidebar.slider('Dilated Kernel size', 1, 49, step=2)
    dilated_iterations = st.sidebar.slider(
        'Dilated Iterations:', 1, 10, step=1)
    selected_img = st.sidebar.selectbox(
        "Apply on:", ["Original image", "Blurred image", "Threshold image", "Canny image"])
    if selected_img.startswith("Canny"):
        dilated_img = cv2.dilate(
            canny_img, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    elif selected_img.startswith("Original"):
        dilated_img = cv2.dilate(
            opencv_image_rgb, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)       
    elif selected_img.startswith("Threshold"):
        dilated_img = cv2.dilate(
            thrsh_img, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    else:
        dilated_img = cv2.dilate(
            blurred_img, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
        
    st.write("""### Dilated image:""")
    st.code("""
    dilated_img = cv2.dilate(image, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    """, language='python')
    st.image(dilated_img)

    # Erroded image
    st.sidebar.subheader("Erroded image")
    erroded_kernel = st.sidebar.slider('Erroded Kernel size', 1, 49, step=2)
    erroded_iterations = st.sidebar.slider(
        'Erroded Iterations:', 1, 10, step=1)
    erroded_img = cv2.erode(
        dilated_img, (erroded_kernel, erroded_kernel), iterations=erroded_iterations)
    st.write("""### Erroded image:""")
    st.code("""
    erroded_img = cv2.dilate(canny_img, (erroded_kernel, erroded_kernel), iterations=erroded_iterations)
    """, language='python')
    st.image(erroded_img)

    
