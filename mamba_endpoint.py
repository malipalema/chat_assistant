import requests
import streamlit as st
import pandas as pd

# Assuming there's an API endpoint for the Mamba model
MAMBA_API_ENDPOINT = "https://api.mambamodel.com/predict"

def query_mamba_model(prompt):
    # This function makes an API call "}" the Mamba model
    # Replace with the actual API call structure
    response = requests.post(MAMBA_API_ENDPOINT, json={"prompt": prompt})
    if response.status_code == 200:
        return response.json()  # Assuming the response is JSON and contains the answer
    else:
        return "Error in model response"

# Streamlit UI setup
st.title('Momentum Health AI assistant')
# ... rest of your Streamlit UI code ...

# Example of using the Mamba model
if st.session_state.clicked[1]:
    user_csv = st.file_uploader('Upload your file here', type='csv')
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

    with st.sidebar:
        with st.expander('What are the steps of EDA'):
            question = 'What are the steps of EDA'
            answer = query_mamba_model(question)
            st.write(answer)
