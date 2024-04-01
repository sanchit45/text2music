from model import model
import scipy

musicgen_processor= model.load_processor()
musicgen_model = model.load_model()

def text_prompt_preprocessing(text: str):
    inputs = musicgen_processor(
    text=text,
    padding=True,
    return_tensors="pt",
    )

    return inputs

def music_generation(preprocessed_text_input):
    audio_values = musicgen_model.generate(**preprocessed_text_input, max_new_tokens=256)
    
    return audio_values

def save_musicfile(music_values):
    sampling_rate = musicgen_model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("..\..\Music\musicgen_out.wav", rate=sampling_rate, data=music_values[0, 0].numpy())
