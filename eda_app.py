import os
#from apikey import apikey
import streamlit as st
import pandas as pd
import requests
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

from lida import Manager, TextGenerationConfig , llm  
#from dotenv import load_dotenv

from PIL import Image
from io import BytesIO
import base64
from langchain import LLMChain
from langchain import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler


#OpenAI key
#os.environ['OPENAI_API_KEY'] = apikey

import boto3script

import time
import sagemaker, boto3script, json
#from cohere_sagemaker import Client
from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base
from typing import Any, Dict, List, Optional
from langchain.embeddings import SagemakerEndpointEmbeddings
#from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from sagemaker.jumpstart.model import JumpStartModel

sagemaker_session = Session()
aws_role = sagemaker_session.get_caller_identity_arn()
aws_region = boto3script.Session().region_name
sess = sagemaker.Session()
#model_version = "*"
model_id, model_version = "meta-textgeneration-llama-2-7b", "2.*"


model = JumpStartModel(model_id=model_id, model_version=model_version)
model_predictor = model.deploy()

payload = {
    "text_inputs": "Tell me the steps to make a pizza",
    "max_length": 50,
    "num_return_sequences": 3,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
}

import json
client = boto3script.client("runtime.sagemaker")
encoded_payload = json.dumps(payload).encode('utf-8') #JSON serialization
response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_payload
    )
model_predictions = json.loads(response["Body"].read())
model_predictions['generated_texts'][0]


from langchain.prompts import PromptTemplate

# In this instance we are just passing in the question for the prompt for our chain
prompt_template = """{question}"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["question"]
)

from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs}).encode('utf-8')
        return input_str

    def transform_output(self, output: str) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]

content_handler = ContentHandler()

model_params = {"max_length": 100,
                "num_return_sequences": 1,
                "top_k": 100,
                "top_p": .95,
                "do_sample": True}

llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name="us-east-1",
        model_kwargs=model_params,
        content_handler=content_handler,
    )

chain = LLMChain(
llm=llm, prompt=prompt)

# Execute chain
sample_prompt = "Tell me the steps to make a pizza"
chain.run(sample_prompt)





# Set the title of the app and page layout
st.set_page_config(page_title='Momentum Health AI Assistant', layout='wide')

# Custom CSS to mimic the Momentum website's style
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
st.title('Momentum Health AI assistant')
#st.header('')
#st.subheader('')
st.write('Welcome to Momentum health\'s chat assistant')

menu = st.sidebar.selectbox("Choose an Option", ["EDA", "Summarize", "Question based Graph"])

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

    

from lida import Manager, TextGenerationConfig # , llm

#model_name = "uukuguy/speechless-llama2-hermes-orca-platypus-13b"
#model_details = [{'name': model_name, 'max_tokens': 2596, 'model': {'provider': 'openai', 'parameters': {'model': model_name}}}]
# assuming your vllm endpoint is running on localhost:8000







text_gen = llm(provider="hf",  api_base="http://localhost:8501/v1")#, api_key="EMPTY")#, models=model_details)
lida = Manager(text_gen = text_gen)
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

def save_image(base64_str, save_path):
    img = base64_to_image(base64_str)
    img.save(save_path)
    print(f"Image saved at {save_path}")

if menu == "Summarize":
    st.subheader("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)
        st.write(summary)
        goals = lida.goals(summary, n=2, textgen_config=textgen_config)
        for goal in goals:
            st.write(goal)
        i = 0
        library = "seaborn"
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        st.image(img)
        
        
elif menu == "Question based Graph":
    st.subheader("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename1.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        text_area = st.text_area("Query your Data to Generate Graph", height=200)
        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                lida = Manager(text_gen = llm("openai")) 
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
                user_query = text_area
                charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
                charts[0]
                image_base64 = charts[0].raster
                img = base64_to_image(image_base64)
                st.image(img)

        




elif menu == "EDA":
    
    #initialize key in session state
    if 'clicked' not in st.session_state:
        st.session_state.clicked={1:False}

    #function to update the value in session state
    def clicked(button):
        st.session_state.clicked[button]=True

    st.button("Start", on_click = clicked, args=[1])
    if st.session_state.clicked[1]:
        st.header("Data Analytics")
        st.subheader("Solution")

        user_csv = st.file_uploader('Upload your file here', type='csv')
        if user_csv is not None:
            user_csv.seek(0)
            df = pd.read_csv(user_csv, low_memory = False)


            #llm = OpenAI(temperature=0)
            @st.cache_data
            def steps_eda():
                steps_eda = llm('What are the steps of EDA')
                return steps_eda

            pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)
            def function_agent():
                st.write("**Data overview**")
                st.write("The first rows of your table look like:")
                st.write(df.head())
                st.write("**Data cleaning**")
                columns_df = pandas_agent.run("What are the meaning of the columns?")
                st.write(columns_df)
                missing_values = pandas_agent.run("How many missing values does this table have?")
                st.write(missing_values)
                duplicates = pandas_agent.run("Are there any duplicates in this table have?")
                st.write(duplicates)
                st.write("**Data Summarizations**")
                st.write(df.describe())
                #correlation_analysis = pandas_agent.run("Calculate correlations between values")
                #st.write(correlation_analysis)
                outliers = pandas_agent.run("Identify outliers")
                st.write(outliers)
                #new_features = pandas_agent.run("Identify outliers")
                #st.write(new_features)
                return

            @st.cache_data
            def function_question_variable():
                st.line_chart(df, y = [user_question_variable])
                summary_statistics = pandas_agent.run(f"Give me a summmary of the variables")
                st.write(summary_statistics)
                normality = pandas_agent.run(f'Check normality of variables')
                st.write(normality)
                outliers = pandas_agent.run(f'Assess the presence of outliers')
                st.write(outliers)
                trends = pandas_agent.run(f'Analyze trends in the data')
                st.write(trends)
                
                return


            @st.cache_data
            def function_question_dataframe():
                dataframe_info = pandas_agent.run(user_question_dataframe)
                st.write(dataframe_info)
                return

    #main
            st.header("EDA")
            st.subheader("General information about your data")

            with st.sidebar:
                with st.expander('What are the steps of EDA'):
                    st.write(steps_eda)

            function_agent()
            st.subheader('Variable of study')
            user_question_variable = st.text_input("Ask questions about your data here?")
            if user_question_variable is not None and user_question_variable != "":
                function_question_variable()

                st.subheader('Further Study')
            if user_question_variable:
                user_question_dataframe = st.text_input("Is there anything else you want to add?")
                if user_question_dataframe is not None and user_question_dataframe not in ("", "no", "No"):
                    function_question_dataframe()
                if user_question_dataframe == ("no", "No"):
                    st.write("")

