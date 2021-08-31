from pathlib import WindowsPath
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pytesseract


st.set_page_config(layout="wide")
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 220px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -220px;
    }
    </style>
    """,
    unsafe_allow_html=True,)

st.markdown("Application Developed by Sai Tej ! 🚀")

st.subheader("""OpenCV  methods""")



uploaded_file = st.file_uploader("Choose a image file", type=['jpg', 'jpeg', 'png', 'jfif','PNG','JPEG','JPG'])
                                
# Useful methods in opencv
st.sidebar.header("Select the methods on your image")


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
    st.sidebar.markdown("Color conversions")
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
    # st.code("""
    #         colored_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    #         """, language='python')
    
    opencv_image_gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    st.write('')
    # Gaussian blur
    #opencv_image_rgb = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)    
    st.subheader("Gaussian Blur")
    p1, p2 = st.columns(2)
    with p1:
        selected_kernel_size = st.slider('Kernel size', 1, 49, step=2)
        kernel_size = (selected_kernel_size, selected_kernel_size)
        blurred_img = cv2.GaussianBlur(
            opencv_image_gray, kernel_size, cv2.BORDER_DEFAULT)
    #st.write("""### Gaussian Blur image:""")
        st.write(f"Kernel size: {(kernel_size)}")
        # st.code("""
        # blurred_img = cv2.GaussianBlur(opencv_image_rgb, kernel_size, cv2.BORDER_DEFAULT)
        # """, language='python')
        st.image(blurred_img)

    # im_border = cv2.copyMakeBorder(
    #     opencv_image_rgb, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # st.image(im_border)
    # Threshold (default Binary)
    
    # opencv_image_rgb = cv2.cvtColor(opencv_image_rgb, cv2.COLOR_BGR2GRAY)
    # blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    st.write('')
    st.subheader("Inverse Thresholding and Noise Removal")
    d1, d2, d3 = st.columns(3)
    
    with d1:        
        threshold_min = st.slider("Threshold_min", 0, 255, step=1, value=100)
        selected_image_type = st.selectbox(
            "Apply on:", ["Grayscale image", "Blurred image"])
    with d2:
        threshold_max = st.slider("Threshold_max", 0, 255, step=1, value=255)
        #Block_size = st.sidebar.slider("Block_size", 0, 50, step=1, value=11)
    with d3:
        threshold_kernal = st.slider("Threshold_Kernal", 1, 100, 50)
        
        
       
    if selected_image_type.startswith("Grayscale"):
        _, thrsh_img = cv2.threshold(
            opencv_image_gray, threshold_min, threshold_max, cv2.THRESH_BINARY_INV)
    #    thrsh_img= cv2.adaptiveThreshold(
    #         colored_img, threshold_max, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Block_size, 2)
                        
    else:
        _, thrsh_img = cv2.threshold(
            blurred_img, threshold_min, threshold_max, cv2.THRESH_BINARY_INV)
        # thrsh_img= cv2.adaptiveThreshold(
        #     blurred_img, threshold_max, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Block_size, 2)
            
            
        #st.write("""### Inverse Thresholding :""")
        # st.code("""
        # thrsh_img = cv2.threshold(opencv_image_rgb, threshold_min, threshold_max, cv2.THRESH_BINARY)
        # """, language='python')
    t1, t2= st.columns(2)
    with t1:
        st.markdown("""### Inverse Thresholding """)
        st.image(thrsh_img)
    
    
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thrsh_img,None, None, None, 8, cv2.CV_32S)
    sizes=stats[1:, -1]
    
    noise_dt=np.zeros((labels.shape),np.uint8)
    
    for i in range(0, nlabels-1):
        if sizes[i] >= threshold_kernal:
            noise_dt[labels == i + 1] = threshold_max
    
    clean_image=cv2.bitwise_not(noise_dt)
    with t2:
        st.markdown("""### Post Noise Removal""")
        st.image(clean_image)
    

    # Canny edges
    st.subheader("Canny edges")
    g1, g2, g3 = st.columns(3)
    with g1:
        threshold1 = st.slider("Threshold 1", 1, 500, step=1, value=300)
        selected_canny_img = st.selectbox(
            "Apply on:", ["Grayscale image", "Blurred image", "Threshold image","Clean Image"])
    with g2:
        threshold2 = st.slider("Threshold 2", 1, 500, step=1, value=200)
    
    if selected_canny_img.startswith("Grayscale"):
        canny_img = cv2.Canny(opencv_image_gray, threshold1, threshold2)
    elif selected_canny_img.startswith("Threshold"):
        canny_img = cv2.Canny(thrsh_img, threshold1, threshold2)
    elif selected_canny_img.startswith("Clean"):
        canny_img = cv2.Canny(clean_image, threshold1, threshold2)            
    else:
        canny_img = cv2.Canny(blurred_img, threshold1, threshold2)
        # st.code("""
    # canny_img = cv2.Canny(opencv_image_rgb, threshold1, threshold2)
    # """, language='python')
    st.image(canny_img)
       
    
       

    # dilated image
    st.subheader("Dilated image")
    k1, k2, k3 = st.columns(3)
    with k1:
        dilated_kernel = st.slider('Dilated Kernel size', 1, 49, step=2)
        selected_img = st.selectbox(
            "Apply on:", ["Grayscale image", "Blurred image", "Threshold image", "Canny image", "Clean image"])
    with k2:
        dilated_iterations = st.slider(
            'Dilated Iterations:', 1, 10, step=1)
        
    
    if selected_img.startswith("Canny"):
        dilated_img = cv2.dilate(
            canny_img, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    elif selected_img.startswith("Grayscale"):
        dilated_img = cv2.dilate(
            opencv_image_gray, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    elif selected_img.startswith("Threshold"):
        dilated_img = cv2.dilate(
            thrsh_img, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    elif selected_img.startswith("Clean"):
        dilated_img = cv2.dilate(
            clean_image, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    else:
        dilated_img = cv2.dilate(
            blurred_img, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
        
    # st.write("""### Dilated image:""")
    # st.code("""
    # dilated_img = cv2.dilate(image, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    # """, language='python')
    st.image(dilated_img)

    # Erroded image
    st.subheader("Erroded image")
    l1, l2, l3 = st.columns(3)
    with l1:
        erroded_kernel = st.slider('Erroded Kernel size', 1, 49, step=2)
    with l2:
         erroded_iterations = st.slider(
        'Erroded Iterations:', 1, 10, step=1)
    erroded_img = cv2.erode(
        dilated_img, (erroded_kernel, erroded_kernel), iterations=erroded_iterations)
    st.write("""### Erroded image:""")
    # st.code("""
    # erroded_img = cv2.dilate(canny_img, (erroded_kernel, erroded_kernel), iterations=erroded_iterations)
    # """, language='python')
    st.image(erroded_img)
    
    st.subheader("OCR Extraction")
    
    selected_ocr_img = st.selectbox(
        "Apply on:", ["Clean Image", "Grayscale image", "Blurred image", "Threshold image"])
    
    che=st.checkbox("Do you Really want OCR Extraction ??", value=False)
    if che:      
        
        if selected_ocr_img.startswith("Clean"):
            final_img = clean_image
        elif selected_ocr_img.startswith("Blurred"):
            final_img = blurred_img
        elif selected_ocr_img.startswith("Threshold"):
            final_img = thrsh_img
        elif selected_ocr_img.startswith("opencv_image_gray"):
            final_img = thrsh_img
        else:
            final_img=erroded_img
        
        ocr_result = pytesseract.image_to_string(final_img, lang='eng')
        st.write("Ohh Crappy results !!")
        l11, l21 = st.columns(2)
        with l11:
            st.image(final_img)
        with l21:
            st.write(ocr_result)
