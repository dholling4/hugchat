import streamlit as st
from hugchat_api import HuggingChat
from hugchat_api.utils import formatHistory, formatConversations

st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")
st.title('🤗💬 HugChat App')

# Hugging Face Credentials
with st.sidebar:
    st.header('Hugging Face Login')
    EMAIL = st.text_input('Enter E-mail:', type='password')
    PASSWD = st.text_input('Enter password:', type='password')
    COOKIE_STORE_PATH = "./usercookies"
    HUG = HuggingChat(max_thread=1)
    # initialize sign in funciton
    sign = HUG.getSign(EMAIL, PASSWD)   
    cookies = sign.login(save=True, cookie_dir_path=COOKIE_STORE_PATH)

# Store AI generated responses
if "messages" not in st.session_state.keys():
    st.session_state['messages'] = [{"role": "assistant", "content": "I'm HugChat, How may I help you?"}]

# Display existing chat messages
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

# If last message is not from assistant, we need to generate a new response
if st.session_state['messages'][-1]["role"] != "assistant":
    # Call LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create ChatBot
            bot = HUG.getBot(email=EMAIL, cookies=cookies)
            # create a new conversation
            conversation_id = bot.createConversation()
            # chat
            message = bot.chat(
                text="hi",
                conversation_id=conversation_id,
                max_tries=2,
                # callback=(bot.updateTitle, (conversation_id,))
            )
            response = message.getFinalText()
            st.write(response)

    #message = {"role": "assistant", "content": response}
    #st.session_state.messages.append(message)

st.write(message)
