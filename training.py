from transformers import AutoTokenizer
from model import Config

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
cfg = Config(d_vocab=len(tokenizer))