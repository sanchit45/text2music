from fastapi import FastAPI
from schema import text_prompt
from utils import text_prompt_preprocessing, music_generation, save_musicfile

app = FastAPI()

@app.post("/text2music")
def text2music(prompt: text_prompt):
    text_input = prompt.text
    processed_text_input= text_prompt_preprocessing(text_input)

    music= music_generation(processed_text_input)

    save_musicfile(music)

    
    return {"response": "Generated"}

