import os
import json
import base64
import asyncio
import chainlit as cl

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

# Configurar la API key de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


@cl.on_chat_start
def start():
    cl.user_session.set("chat", ChatOpenAI(model="gpt-4o", temperature=0))


async def stream_response(content):
    message = cl.Message(content="")
    await message.send()

    chunk_size = 100  # Ajusta este valor para controlar el tamaño de los chunks
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        await message.stream_token(chunk)
        await asyncio.sleep(0.05)  # Ajusta este valor para controlar la velocidad de "tecleo"

    await message.update()


def add_ids_and_names(json_data, id_counter=1):
    if isinstance(json_data, dict):
        if 'name' in json_data:  # Presuponiendo que la IA ya identifica un campo "name" con el identificador visual
            json_data['id'] = id_counter
            id_counter += 1
        for key, value in json_data.items():
            if isinstance(value, (dict, list)):
                id_counter = add_ids_and_names(value, id_counter)
    elif isinstance(json_data, list):
        for item in json_data:
            id_counter = add_ids_and_names(item, id_counter)
    return id_counter


@cl.on_message
async def main(message: cl.Message):
    chat = cl.user_session.get("chat")

    if message.elements:
        image = message.elements[0]
        if image.mime.startswith("image/"):
            image_path = image.path
            base64_image = encode_image(image_path)

            messages = [
                SystemMessage(
                    content="""
                        You are an AI specialized in analyzing data center floor plans. 
                        Your task is to examine the provided image and generate a structured JSON representation 
                        of the data center layout, including rooms, rows, and racks. 
                        It's crucial to identify and include all cooling units (AC or AHU) as racks with type 'REF'.
                        Associate each cooling unit with the nearest row and include them in the 'racks' list of that row.
                        Do not create a separate section for cooling units.
                        Each element should have an 'id' which is an auto-incrementing number, and a 'name' that 
                        corresponds to the identifier visible in the image.
                    """
                ),
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": """
                            Analyze this data center floor plan and provide a structured JSON 
                            representation of the layout. Include rooms, rows, and racks. 
                            Use 'type' field with values 'IT' for IT racks and 'REF' for cooling units.
                            Make sure to identify all AC or AHU units and include them as racks of type 'REF' 
                            in the 'racks' list of the appropriate rows. Do not create a separate section for cooling units.
                            Respond only with the JSON structure, do not include any explanatory text.
                            Each element should have an 'id' which is an auto-incrementing number, and a 'name' that 
                            corresponds to the identifier visible in the image.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ])
            ]

            loading_message = cl.Message(content="Procesando el floor plan...")
            await loading_message.send()

            response = await cl.make_async(chat.invoke)(messages)

            await loading_message.remove()

            try:
                json_structure = json.loads(response.content)

                # Agregar IDs autonuméricos y utilizar el nombre o identificador del JSON generado
                add_ids_and_names(json_structure)

                formatted_json = json.dumps(json_structure, indent=2)
                cl.user_session.set("last_json", json_structure)  # Guardamos el JSON para uso posterior
                await stream_response(f"Aquí está la estructura JSON del floor plan:\n```json\n{formatted_json}\n```")
            except json.JSONDecodeError as je:
                print('JSONDecodeError', str(je))
                await stream_response(response.content)
        else:
            await cl.Message(content="Por favor, sube una imagen válida del floor plan.").send()
    else:
        await cl.Message(
            content="Por favor, sube una imagen del floor plan para analizarla.").send()


@cl.on_chat_end
def end():
    print("Chat ended")


@cl.action_callback("image_upload")
async def on_image_upload(action):
    await main(cl.Message(content="", elements=action.files))
