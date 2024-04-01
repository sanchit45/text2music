import transformers
from transformers import AutoProcessor, MusicgenForConditionalGeneration
    
class musicgen():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(musicgen, cls).__new__(cls)
        return cls._instance
    
    def load_processor(cls):
        return AutoProcessor.from_pretrained("facebook/musicgen-small")
    
    def load_model(cls):
        return MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    

model= musicgen()


