import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import google.generativeai as genai

# Config function
st.set_page_config(page_title='Dr Sajad AI Bot', page_icon=None, layout='centered', initial_sidebar_state='auto')

# Hiding the header and footer
hide_menu_style="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_menu_style, unsafe_allow_html=True)

api_key=st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model_gene = genai.GenerativeModel('gemini-1.5-flash')

# Define background images for different pages
background_images = {
    "Home": "https://img.freepik.com/free-photo/portrait-concentrated-male-doctor-dressed-uniform_171337-1495.jpg?w=900&t=st=1720435979~exp=1720436579~hmac=0a617a10f4dee02cc0080558c56a10d54bf984b2d57f2ad96b3e087c048f5465",
    "Radiologist": "https://img.freepik.com/free-vector/abstract-medical-wallpaper-template-design_53876-61804.jpg?t=st=1720439409~exp=1720443009~hmac=88c927a6021f5e3134288a804a32fe7553c8871510372478a9af380087ba68d8&w=826",
    "Contact": "https://img.freepik.com/free-photo/portrait-professional-doctor-holding-chest-x-ray-shot-looking-camera_1098-19302.jpg?w=740&t=st=1720436074~exp=1720436674~hmac=e9986f8e3f0c0c151f5e06dbf9ac5b306854aa3f542860642efdd87eed99878a",
}

#------------------ Inject CSS for background image and layout adjustments -------------------------------------
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 5px dashed #150E0D ; /* Change border style */
            # padding: 20px; /* Add padding inside the drop zone */
            border-radius: 20px; /* Optional: Rounded corners */
            background-color: #f9f9f9; /* Background color */
            # text-align: center; /* Center-align content */
            margin-top: -50px; /* Adjust margin to reduce space above */
            font-weight: bold
            }


        [data-testid="baseButton-secondary"]{
            font-weight:bold
            }

         [data-testid="stImage"] {
            border: 5px solid #161212; /* Change border color to blue */
            border-radius: 15px; /* Optional: Rounded corners */
            padding: 3px; /* Add padding inside the border */
            background-color: #FFFFFF; /* Background color behind the image */
            box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.1); /* Optional: Add a shadow effect */
            margin-left:200px;
            margin-top:-20px

        }
        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
            height: 0px; /* Optional: Adjust the height if necessary */
        }
        
        .uploader-text {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            text-align:center;

            width: 52vw; /* Use full viewport width */
           margin-bottom:-20px;

           margin-top:-60px

        }

        .custom-text {
        color: #A02020; /* Change text color to blue */
        font-size: 24px; /* Increase font size */
        font-weight: bold; /* Make the text bold */
        text-align: center; /* Center align the text */
        margin: 20px 0; /* Add margin for spacing */
        }
        .custom-header {
            background-color: rgba(0, 0, 0, 0); /* Optional: Add a semi-transparent overlay */
            padding: 0px;
            text-align: center;
            width: 60vw; /* Use full viewport width */
            color: white; /* Default text color */
            position: fixed;
            top: 0;
            
        }
        .custom-header h1 {
            color: #F4F3F3; /* Default text color */
        }

        .custom-header-radiologist {
            background-color: rgba(0, 0, 0, 0); /* Optional: Add a semi-transparent overlay */
            padding: 0px;
            text-align: center;
            width: 55vw; /* Use full viewport width */
            color: white; /* Default text color */
            # position: fixed;
            margin-bottom:-70px;
            margin-Top:-130px;
            top: 0;

            z-index: 1000;
        }
        .custom-header-radiologist h1 {
            color: #150E0D; /* Default text color */
        }
        
        .custom-header-doctor {
            background-color: rgba(0, 0, 0, 0); /* Optional: Add a semi-transparent overlay */
            padding: 0px;
            text-align: center;
            width: 55vw; /* Use full viewport width */
            color: white; /* Default text color */
            # position: fixed;
            margin-bottom:-70px;
            margin-Top:-130px;
            top: 0;

            z-index: 1000;
        }
        .custom-header-doctor h1 {
            color: #150E0D; /* Default text color */
        }
        .stApp {
            padding-top: 50px; /* Adjust based on your custom header height */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar selection for page
selected_page = st.sidebar.radio("Select Page", ["Home", "Radiologist", "Contact"])

# Conditional background image and header rendering based on selected page
if selected_page in background_images:
    st.markdown(
        f"""
        <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{background_images[selected_page]}");
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Conditional header rendering based on selected page

# -------------------------------------------- Home Page -------------------------------------------------
if selected_page == "Home":
    st.markdown('<div class="custom-header"><h1>ToothLens: X-Ray Diagnosis & Treatment</h1></div>', unsafe_allow_html=True)



# ------------------------------------ Radiologist Page -----------------------------------------------------
elif selected_page == "Radiologist":
    st.markdown('<div class="custom-header-radiologist"><h1>Periapical X-ray Radiologist</h1></div>',
                unsafe_allow_html=True)

    # # Function for loading Trained Model
    # @st.cache_resource
    # def load_model():
    #     model = tf.keras.models.load_model(r'model.h5')
    #     return model


    # with st.spinner('Model is being loaded..'):
    #     model = load_model()

    # st.markdown('<p class="uploader-text">Upload your Periapical X-ray (JPG/PNG only)</p>', unsafe_allow_html=True)
    # file = st.file_uploader("", type=["jpg", "png"])

    # st.set_option('deprecation.showfileUploaderEncoding', False)

    # # Function for converting image to 3D and reshaping for model input
    # def import_and_predict(image_data, model):
    #     size = (256, 256)
    #     image = ImageOps.fit(image_data, size, Image.BILINEAR)
    #     image = np.asarray(image)


    #     # Ensure the image is grayscale
    #     if image.ndim == 2:  # Image is grayscale
    #         # Convert grayscale to RGB by duplicating the single channel
    #         image = np.stack((image,) * 3, axis=-1)
    #     elif image.shape[2] == 1:  # Image has a single channel
    #         # Convert single channel to RGB
    #         image = np.concatenate([image] * 3, axis=-1)

    #     # Normalize the image
    #     image = image / 255.0

    #     img_reshape = image[np.newaxis, ...]

    #     prediction = model.predict(img_reshape)

    #     return prediction


    # if file is None:
    #     st.text("")
    # else:

    #     image = Image.open(file)
    #     st.image(image, width=300)

    #     predictions = import_and_predict(image, model)

    #     class_names = ['Primary Endo with Secondary Perio', 'Primary Endodontic Lesion',
    #                    'Primary Perio with Secondary Endo', 'Primary Periodontal Lesion', 'True Combined Lesions']

    #     # Determine the detected class
    #     detected_class = class_names[np.argmax(predictions)]

    #     st.markdown(f'<p class="custom-text">The Lesion detected is :  {detected_class}</p>', unsafe_allow_html=True)


# ------------------------------------- Doctor Sajad AI Bot Page --------------------------------------------------------------------------
elif selected_page == "Contact":
    st.markdown('<div class="custom-header-doctor"><h1>Doctor Sajad</h1></div>', unsafe_allow_html=True)


    persona = """
            You are Doctor Muhammad Sajad AI bot. You help people answer questions about yourself (i.e. Doctor Muhammad Sajad). Answer as if you are responding. Don't answer in the second or third person. If you don't know the answer, simply say "That's a secret."

            Here is more info about Doctor Muhammad Sajad:

            Muhammad Sajad is a dental professional specializing in endodontics and periodontics. With 15 years of clinical practice, I have extensive experience in managing complex dental lesions involving both the tooth pulp and surrounding periodontal structures. I completed my Doctor of Dental Surgery (DDS) from the University of Michigan and am board-certified in both endodontics and periodontics. I practice at SmileCare Dental Clinic in New York, NY.

            I am known for a patient-centered approach, combining advanced diagnostic techniques with evidence-based treatments.

            If some one asked about the prescription of given teeth lesions I am have to write the prescription in following manner.
            Prescriptions for Teeth Lesions:

            1. **Primary Endo with Secondary Perio**
               - **Diagnosis:** Root canal infection leading to secondary periodontal involvement.
               - **Prescription:**
                 - **Endodontic Treatment:** 
                   - Root canal therapy to remove the infected pulp tissue.
                   - Intracanal medication with calcium hydroxide for 1-2 weeks.
                   - Final obturation of the root canal system.
                 - **Periodontal Treatment:**
                   - Scaling and root planing.
                   - Antibiotic therapy (e.g., Amoxicillin 500 mg, TID for 7 days).
                   - Chlorhexidine mouth rinse (0.12%) twice daily for 2 weeks.

            2. **Primary Endodontic Lesion**
               - **Diagnosis:** Infection or necrosis confined to the pulp chamber and root canals.
               - **Prescription:**
                 - **Endodontic Treatment:**
                   - Root canal therapy to remove the infected pulp tissue.
                   - Intracanal medication with calcium hydroxide if needed.
                   - Final obturation of the root canal system.
                 - **Pain Management:**
                   - NSAIDs (e.g., Ibuprofen 600 mg, TID for 3 days).

            3. **Primary Perio with Secondary Endo**
               - **Diagnosis:** Periodontal disease extending to the apex, causing secondary endodontic involvement.
               - **Prescription:**
                 - **Periodontal Treatment:**
                   - Scaling and root planing.
                   - Local delivery of antibiotics (e.g., doxycycline gel).
                 - **Endodontic Treatment:**
                   - Root canal therapy to manage the secondary endodontic involvement.
                 - **Systemic Antibiotic Therapy:**
                   - Amoxicillin 500 mg, TID for 7 days or Metronidazole 400 mg, TID for 7 days if allergic to penicillin.
                 - **Chlorhexidine Mouth Rinse:**
                   - 0.12% solution, twice daily for 2 weeks.

            4. **Primary Periodontal Lesion**
               - **Diagnosis:** Disease affecting the periodontal tissues without pulpal involvement.
               - **Prescription:**
                 - **Periodontal Treatment:**
                   - Scaling and root planing.
                   - Local delivery of antibiotics (e.g., minocycline microspheres).
                 - **Systemic Antibiotic Therapy:**
                   - Amoxicillin 500 mg, TID for 7 days or Doxycycline 100 mg, once daily for 10 days.
                 - **Chlorhexidine Mouth Rinse:**
                   - 0.12% solution, twice daily for 2 weeks.

            5. **True Combined Lesions**
               - **Diagnosis:** Concurrent pulpal and periodontal disease without a clear primary source.
               - **Prescription:**
                 - **Endodontic Treatment:**
                   - Root canal therapy to address the pulpal infection.
                 - **Periodontal Treatment:**
                   - Scaling and root planing.
                   - Surgical intervention if necessary (e.g., flap surgery).
                 - **Systemic Antibiotic Therapy:**
                   - Amoxicillin 500 mg, TID for 7 days.
                 - **Chlorhexidine Mouth Rinse:**
                   - 0.12% solution, twice daily for 2 weeks.
                 - **Follow-up:** Regular periodontal maintenance and monitoring of endodontic healing.

            If someone wants an appointment, they can contact me via email: muhammad_sajad47@yahoo.com.
            """

    st.title(" Dotor Sajad's AI Bot")
    st.write("")
    # st.write("Ask anything about me")
    user_question = st.text_input("Ask anything about me", placeholder="Type your question here")
    if st.button("ASK", use_container_width=400):
        prompt = persona + " Here is the question that the user asked: " + user_question
        response = model_gene.generate_content(prompt)
        st.write(response.text)
