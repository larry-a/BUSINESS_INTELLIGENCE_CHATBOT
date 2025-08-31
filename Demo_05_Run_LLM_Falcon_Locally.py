import streamlit as st
import torch
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

st.title("Falcon LLM")

# Load model (cached)
@st.cache_resource
def load_falcon():
    MODEL_NAME = "tiiuae/falcon-7b-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, load_in_8bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    # Generation config
    generation_config = model.generation_config
    generation_config.temperature = 0.9
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 256
    generation_config.use_cache = False
    generation_config.repetition_penalty = 1.7
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    # Create pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    llm_pipeline = HuggingFacePipeline(pipeline=pipe)

    # Prompt template
    new_template = """
The following is a conversation between a human and an AI. The AI behaves like Albert Einstein, providing detailed explanations about physics.

Current conversation:
{history}
Human: {input}
AI:""".strip()

    prompt = PromptTemplate(input_variables=["history", "input"], template=new_template)
    memory = ConversationBufferWindowMemory(memory_key="history", k=6, return_only_outputs=True)
    chain = ConversationChain(llm=llm_pipeline, memory=memory, prompt=prompt, verbose=False)
    
    return chain

# Initialize
if 'chain' not in st.session_state:
    st.session_state.chain = load_falcon()

# User input
text = st.text_input("Ask a question:")

if text:
    res = st.session_state.chain.predict(input=text)
    st.write(res)
