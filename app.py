# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github

'''
git init
git add .
git remote add origin https://github.com/nicoig/omnibot.git
git commit -m "Initial commit"
git push -u origin master

'''

# Actualizar Repo de Github
#git add .
#git commit -m "Se actualizan las variables de entorno"
#git push origin master


# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12


################################################

import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO

# Cargar la clave de la API desde el archivo .env
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Función para consultar y transmitir la respuesta del LLM
def stream_llm_response(client, model_params):
    response_message = ""

    for chunk in client.chat.completions.create(
        model=model_params.get("model", "o1-mini"),
        messages=st.session_state.messages,
        temperature=model_params.get("temperature", 0.3),
        max_tokens=4096,
        stream=True,
    ):
        content = chunk.choices[0].delta.content
        if content:
            response_message += content
            yield content

    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}]
    })

# Función para convertir una imagen a base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def main():
    st.set_page_config(
        page_title="El OmniChat",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown("<h1 style='text-align: center; color: #6ca395;'>🤖 <i>El OmniChat</i> 💬</h1>", unsafe_allow_html=True)

    client = OpenAI(api_key=OPENAI_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"])

    with st.sidebar:
        st.divider()
        model = st.selectbox("Selecciona un modelo:", [
            "o1-mini",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-32k",
        ], index=0)

        model_temp = st.slider("Temperatura", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        respuesta_audio = st.checkbox("Respuesta en audio", value=False)
        if respuesta_audio:
            cols = st.columns(2)
            with cols[0]:
                voz_tts = st.selectbox("Selecciona una voz:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            with cols[1]:
                modelo_tts = st.selectbox("Selecciona un modelo:", ["tts-1", "tts-1-hd"], index=1)

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

        def reset_conversation():
            if "messages" in st.session_state and st.session_state.messages:
                st.session_state.pop("messages", None)

        st.button("🗑️ Reiniciar conversación", on_click=reset_conversation)
        st.divider()

        if model in ["gpt-4o-mini"]:
            st.write("### **🖼️ Añadir una imagen:**")

            def add_image_to_messages():
                if st.session_state.uploaded_img or "camera_img" in st.session_state:
                    img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                    raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                    img_base64 = get_image_base64(raw_img)
                    st.session_state.messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_base64}"}}]
                    })

            cols_img = st.columns(2)
            with cols_img[0]:
                st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"], key="uploaded_img", on_change=add_image_to_messages)

            with cols_img[1]:
                activar_camara = st.checkbox("Activar cámara")
                if activar_camara:
                    st.camera_input("Toma una foto", key="camera_img", on_change=add_image_to_messages)

        st.write("### **🎤 Añadir un audio:**")
        audio_prompt = None
        if "prev_speech_hash" not in st.session_state:
            st.session_state.prev_speech_hash = None

        speech_input = audio_recorder("Presiona para hablar:", icon_size="3x", neutral_color="#6ca395")
        if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
            st.session_state.prev_speech_hash = hash(speech_input)
            transcript = client.audio.transcriptions.create(model="whisper-1", file=("audio.wav", speech_input))
            audio_prompt = transcript.text

    if prompt := st.chat_input("¡Hola! Pregúntame lo que quieras...") or audio_prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt or audio_prompt}]
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(client, model_params))

        if respuesta_audio:
            response = client.audio.speech.create(
                model=modelo_tts,
                voice=voz_tts,
                input=st.session_state.messages[-1]["content"][0]["text"],
            )
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
