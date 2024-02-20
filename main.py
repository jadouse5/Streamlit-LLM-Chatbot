import streamlit as st
from streamlit_chat import message
from transformers import pipeline

# Initialize the chat model pipeline from Huggingface
from transformers import AutoTokenizer, MistralForCausalLM

model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Set up Streamlit page configuration
st.set_page_config(page_title="Chat with AI")

# Streamlit layout containers
input_container = st.container()
response_container = st.container()

with st.sidebar:
    st.title('💬 Chat with AI')
    st.markdown('''
    ## About
    This app is powered by a state-of-the-art language model from Huggingface:
    - Model: [OpenAssistant/oasst-sft-6-llama-30b-xor](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor)
    
    💡 Dive into an interactive conversation with AI!
    ''')
    st.write('Developed by Your Name')

# Initialize session state for tracking conversation
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! I'm an AI trained to chat. How can I help you today?"]
if 'past' not in st.session_state:
    st.session_state['past'] = ["Welcome!"]

# User input
with input_container:
    user_input = st.text_input("You: ", key="input")

# Function for generating responses using the Huggingface model
def generate_response(prompt):

    responses = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(responses.input_ids, max_length=30)
    tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    if responses:
        return responses[0]['generated_text']
    else:
        return "I'm not sure how to respond to that."

# Response output and conversation logic
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['generated'][i], key=str(i))
