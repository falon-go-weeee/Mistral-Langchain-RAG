from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from transformers import BitsAndBytesConfig

from rag_redis.config import (
    EMBED_MODEL,
    INDEX_NAME,
    INDEX_SCHEMA,
    REDIS_URL,
)


# Make this look better in the docs.
class Question(BaseModel):
    __root__: str


# Init Embeddings
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Connect to pre-loaded vectorstore
# run the ingest.py script to populate this
vectorstore = Redis.from_existing_index(
    embedding=embedder, index_name=INDEX_NAME, schema=INDEX_SCHEMA, redis_url=REDIS_URL
)
# TODO allow user to change parameters
retriever = vectorstore.as_retriever(search_type="mmr")


# Define our prompt
template = """
Use the following pieces of context from Nike's financial 10k filings
dataset to answer the question. Do not make up an answer if there is no
context provided to help answer it. Include the 'source' and 'start_index'
from the metadata included in the context you used to answer the question

Context:
---------
{context}

---------
Question: {question}
---------

Answer:
"""


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


# RAG Chain

model_name='mistralai/Mistral-7B-Instruct-v0.1'
model_config = transformers.AutoConfig.from_pretrained(
    model_name,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "bfloat16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True, 
    quantization_config=bnb_config, 
    device_map="auto"
)

text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# model = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | mistral_llm
    | StrOutputParser()
).with_types(input_type=Question)
