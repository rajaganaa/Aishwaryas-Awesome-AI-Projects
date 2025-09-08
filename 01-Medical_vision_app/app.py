import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import tempfile
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"An error occurred during API configuration: {e}")
    st.stop()

# The prompt for the vision model
sample_prompt = """You are a medical practictioner and an expert in analzying medical related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

Now analyze the image and answer the above questions in the same structured manner defined above."""

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def call_gemini_vision_for_analysis(filename: str, prompt=sample_prompt):
    """
    Analyzes an image using the Gemini Pro Vision model.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest') 
    try:
        with Image.open(filename) as img:
            response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        return None

def chat_eli_gemini(query):
    """
    Explains a text using the Gemini Pro model (for text-only tasks).
    """
    eli5_prompt = "You have to explain the below piece of information to a five year old. \n" + query
    # THE FINAL FIX: Using the same powerful model for the text-only task.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        response = model.generate_content(eli5_prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred during explanation: {e}")
        return None

# --- Streamlit App UI ---
st.title("Medical Help using Gemini Vision")

with st.expander("About this App"):
    st.write("Upload a medical image to get an analysis from Google's Gemini Vision Pro model.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['filename'] = tmp_file.name
    st.image(uploaded_file, caption='Uploaded Image')

if st.button('Analyze Image'):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        with st.spinner("Gemini is analyzing the image..."):
            st.session_state['result'] = call_gemini_vision_for_analysis(st.session_state['filename'])
        
        os.unlink(st.session_state['filename'])
        
        if st.session_state['result']:
            st.markdown(st.session_state['result'])

if 'result' in st.session_state and st.session_state['result']:
    st.info("Below you have an option for ELI5 to understand in simpler terms.")
    if st.radio("Explain Like I'm 5 (ELI5)", ('No', 'Yes')) == 'Yes':
        with st.spinner("Generating a simpler explanation..."):
            simplified_explanation = chat_eli_gemini(st.session_state['result'])
        if simplified_explanation:
            st.markdown(simplified_explanation)