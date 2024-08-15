import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama3'

    config.tokenizer_paths=["/root/autodl-tmp/llms/Meta-Llama-3-8B-Instruct"]
    config.model_paths=["/root/autodl-tmp/llms/Meta-Llama-3-8B-Instruct"]
    config.conversation_templates=['llama-3']
    config.batch_size=256
    return config