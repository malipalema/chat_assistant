import time
import sagemaker, boto3script, json
#from cohere_sagemaker import Client
from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base
from typing import Any, Dict, List, Optional
from langchain_community.embeddings import SagemakerEndpointEmbeddings
#from langchain.llms.sagemaker_endpoint import ContentHandlerBase

#sagemaker_session = Session()
#aws_role = sagemaker_session.get_caller_identity_arn()
aws_region = boto3script.Session().region_name
#sess = sagemaker.Session()
#model_version = "*"

#print(model_version)


sess = sagemaker.Session()
role = sagemaker.get_execution_role()
print(sess)
print(role)
print(aws_region)


