from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain

# create a prompt template where it contains some initial instructions
# here we say our LLM to think step by step and give the answer

template = """
Let's think step by step of the question: {question}
Based on all the thought the final answer becomes:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# paste the path where your model's weight are located (.bin file)
# you can download the models by going to gpt4all's website.
# scripts for downloading is also available in the later 
# sections of this tutorial

local_path = ("./models/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin")

# initialize the LLM and make chain it with the prompts

llm = GPT4All(
    model=local_path, 
    backend="llama", 
)

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# run the chain with your query (question)

llm_chain('Who is the CEO of Google and why he became the ceo of Google?')
