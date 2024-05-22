import pandas as pd
import streamlit as st
st.set_page_config(page_title='Momentum Health AI Assistant', layout='wide')

from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
from typing import List, Dict
import json, os
import seaborn as sns
from PIL import Image
from io import BytesIO
import base64
#from streamlit_authenticator import Authenticate
import boto3
import yaml
#import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          ResetError,
                                                          UpdateError) 

# Loading config file
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    #config['pre-authorized']
)

# Creating a login widget
try:
    authenticator.login()
except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
    authenticator.logout()
    #st.write(f'Welcome *{st.session_state["name"]}*')
    #st.title('Some content')
    #st.title(f"Welcome {st.session_state['name']} to the application")
# Display based on authentication status
#if st.session_state['authenticated']:
    #st.write("Here is your main dashboard...")
    custom_css = """
    <style>
        html, body, [class*="css"] {
            font-family: "Arial", sans-serif;  # Choose a font similar to the website's
        }
        .stButton>button {
            color: white;
            background-color: #00529B;  # Momentum primary blue color
            border-radius: 10px;
            border: 1px solid #00529B;
        }
        .stTextInput>div>div>input {
            color: #00529B;
        }
        header, footer {
            display: none;
        }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    # Logo and Title (Assuming you have the logo aligned to the right)
    col1, col2, col3 = st.columns([8, 1, 1])
    col2.image("data:image/jpeg;base64,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", width=100)  # Adjust path and width

    # Rest of your Streamlit UI code goes here
    #st.title('Momentum Health AI assistant')
    #st.header('')
    #st.subheader('')
    #st.write('Welcome to Momentum health\'s chat assistant')
    st.title(f"{st.session_state['name']}, Welcome to Momentum health\'s chat assistant")
    # Function to safely parse the code block and execute it
    def execute_code(code, local_vars):
        try:
            # Include the necessary imports directly in the locals() for exec()
            # so they are available when the code block is executed.
            local_imports = {
                'pd': pd,
                'plt': plt,
                'st': st, 
            }
            # Add the 'uploaded_file' to the local_vars if it's not already there
            #if 'uploaded_file' not in local_vars:
            #   local_vars['uploaded_file'] = uploaded_file

            # Merge the dictionaries. The local_vars will have DataFrame `df` that you got from the CSV.
            exec_locals = {**local_imports, **local_vars}

            # Execute the code block with access to the necessary libraries and local vars
            exec(code, {}, exec_locals)

            # Retrieve the figure from the locals dictionary after execution
            fig = exec_locals.get('fig', None)

            # If a figure is created, display it using Streamlit
            if fig:
                st.pyplot(fig)
            else:
                # No figure was found; handle this case as needed
                st.error("No figure was created by the code block.")
        except Exception as e:
            st.error(f"An error occurred while executing the code block: {e}")


    # Define the content handler for the Sagemaker endpoint
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
            # Convert prompt and parameters to JSON and encode as bytes
            input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            # Decode JSON response and extract the generated text
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generated_text"]

    # Assuming you have a function to generate dynamic context from DataFrame
    #def generate_dynamic_context(df):
    ## return f"Columns: {columns}" 

    # Initialize S3 client
    s3 = boto3.client("s3")
    bucket_name = "testbucket-streamlit"

    # Get the list of objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name).get("Contents", [])
    file_names = [obj['Key'] for obj in objects]

    # Select a file from the dropdown
    selected_file = st.selectbox("Choose a file from S3:", file_names)

    # Function to read the selected file from S3
    def load_data_from_s3(bucket, key):
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read()
        return pd.read_csv(BytesIO(data))

    # Load data and display
    if st.button("Load Data"):
        df = load_data_from_s3(bucket_name, selected_file)
        st.write(df)

    # Streamlit UI for uploading a CSV file
    #uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    #if uploaded_file is not None:
       # df = pd.read_csv(uploaded_file)
        #context = generate_dynamic_context(df)
        st.write("The first rows of your table look like:")
        st.write(df.head())
        question = st.text_input("Enter your analysis question:")
        column_names = df.columns.tolist()  # Get the list of column names
        # Create a template with placeholders for the column names
        template1 = f"""
        <s>[INST] <<SYS>>
        You are a helpful code assistant that can teach a junior developer how to code.
        Your language of choice is Python. Don't explain the code, just generate the code block itself.
        Assume data is already loaded into a DataFrame from the user's uploaded CSV and is called 'df'.
        Use the DataFrame columns ({', '.join(column_names)}) to generate code based on the user's question.

        \n\n{{context}}\n\nQuestion: {{question}}
        <</SYS>>
        [/INST]
        """

        # Setup the QA chain
        chain = load_qa_chain(
            llm=SagemakerEndpoint(
                endpoint_name="huggingface-pytorch-tgi-inference-2024-05-06-06-19-04-086",
                region_name="eu-west-1",
                model_kwargs={"temperature": 1, "max_new_tokens": 300},
                content_handler=ContentHandler(),
            ),
            prompt=PromptTemplate(
                template=template1,  # Ensure 'template' is defined earlier as shown in previous examples
                input_variables=["context", "question"]
            ),
        )
            #print(context)
            # Invoke the chain with the dynamically created context and user question
            #document = {"page_content": file_content}  # Adjust this to match the expected format
            #response = chain.invoke({"input_documents": [document], "context": context, "question": question}, return_only_outputs=True)
        if st.button("Generate Code"):
            #file_content = uploaded_file.getvalue().decode("utf-8")
            # Before invoking the chain, replace placeholder in the template with actual column names
            #columns_placeholder = {'column_name': df.columns[0]}  # Example: Choose the first column for the histogram
            # You could modify the above line to select a specific column based on user input or some other logic

            # Replace 'column_name' placeholder in the template with the actual first column name
            #modified_template = template.replace("'column_name'", columns_placeholder['column_name'])
            #response = chain.invoke({"input_documents": uploaded_file, "question": question}, return_only_outputs=True)
            #response_text = response["output_text"]
            # Define the start and end delimiters of the instructional text
            response = chain.invoke({"input_documents": uploaded_file, "question": question}, return_only_outputs=True)
            response_text = response["output_text"]

            # Define the start and end delimiters of the instructional text
            start_delimiter = "<s>[INST] <<SYS>>"
            end_delimiter = "[/INST]"

            # Find the index position of the delimiters
            start_index = response_text.find(start_delimiter)
            end_index = response_text.find(end_delimiter, start_index)

            # Check if both delimiters were found
            if start_index != -1 and end_index != -1:
                # Calculate the end index to include the end delimiter length
                end_index += len(end_delimiter)
                # Extract the text after the end delimiter
                code_block = response_text[end_index:].strip()
            else:
                code_block = response_text  # If delimiters not found, use the whole response

            # Split the code block into lines and dedent
            lines = code_block.split('\n')
            dedented_code = '\n'.join(line.lstrip() for line in lines)

            # Print the dedented code to check it
            print(dedented_code)

            # Display the dedented code in the Streamlit app
            st.code(dedented_code)


            # Prepare local variables for execution, including the dataframe
            local_vars = {
                'pd': pd,
                'plt': plt,
                'df': df,
                'st': st,
                'uploaded_file': uploaded_file
            }
            #if 'uploaded_file' not in local_vars:
                #  local_vars['uploaded_file'] = uploaded_file
            execute_code(dedented_code, local_vars)
            # If a matplotlib figure is returned, display it
            
            st.pyplot(plt.gcf())  # gcf - Get Current Figure

else:
    st.info("Please login to access the application.")

# Registration and password reset are typically handled on different pages or conditional blocks
# Ensure you integrate these functionalities securely and contextually within your application's flow.
#st.set_page_config(page_title='Momentum Health AI Assistant', layout='wide')
menu = st.sidebar.selectbox("Choose an Option", ["Question Answer", "EDA", "Question Answering", "Question based Graph", "Summarize"])

with st.sidebar:
    st.divider()
    st.write('''Why Choose Momentum Health AI 🚀
!''')
    st.caption('''
    🌐 Effortless Efficiency: Say goodbye to complexity. Momentum simplifies tasks, giving you more time for what matters.

    🤖 Tailored Just for You: It's not a one-size-fits-all bot. Momentum adapts to your needs, offering a personalized experience every time.

    🚀 Seamless Integration: No disruptions. No learning curves. Just smooth integration into your workflow.

    ✨ Powered by OpenAI: Cutting-edge technology at your service for accurate and context-aware responses.

    🔐 Security First: Your data's safety is our priority. Momentum complies with the highest industry standards.

    🌟 Always Here for You: Our support team is ready 24/7. Got questions? We've got answers. ''')     

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


def save_image(base64_str, save_path):
    img = base64_to_image(base64_str)
    img.save(save_path)
    print(f"Image saved at {save_path}")

# Additional app logic and display elements
#st.write("Further information and data visualization tools will be here.")
