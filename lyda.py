import os
from apikey import apikey
import streamlit as st
import pandas as pd

#from langchain.llms import OpenAI
#import openai
#rom langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

from dotenv import load_dotenv, find_dotenv

from lida import Manager, TextGenerationConfig , llm  
#from dotenv import load_dotenv

from PIL import Image
from io import BytesIO
import base64
#OpenAI key
os.environ['OPENAI_API_KEY'] = apikey
from lida import Manager, llm

lida = Manager(text_gen = llm("openai")) # palm, cohere ..
summary = lida.summarize("C:\\Users\\MaKhang\\chat assistant\\health-insurance-premiums-on-policies-written-in-new-york-annually-1.csv")
goals = lida.goals(summary, n=2) # exploratory data analysis
charts = lida.visualize(summary=summary, goal=goals[0])