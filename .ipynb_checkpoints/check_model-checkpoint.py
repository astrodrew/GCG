from transformers import AutoModelForCausalLM,AutoTokenizer
from fastchat.model import get_conversation_template
# model = AutoModelForCausalLM.from_pretrained(
#             '/root/autodl-tmp/llms/Meta-Llama-3-8B-Instruct',
#             device_map = 'auto') 
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/llms/Meta-Llama-3-8B-Instruct')
# tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/llms/Llama-2-7b-chat-hf')

# conv_template = get_conversation_template('/root/autodl-tmp/llms/Meta-Llama-3-8B-Instruct')

raw_suffix = '! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !'
tokens = tokenizer(raw_suffix, add_special_tokens=False).input_ids
print(raw_suffix)
print(tokens)
print(len(tokens))

decode_str = tokenizer.decode(tokens)
print(decode_str)

new_tokens = tokenizer(decode_str, add_special_tokens=False).input_ids

print(new_tokens)
print(len(new_tokens))




# tokens = tokenizer.convert_ids_to_tokens(tokens)
# print(tokens)  # 输出: ['[CLS]', 'hello', 'world', '[SEP]']
# # Tokens 转换回文本
# text = tokenizer.convert_tokens_to_string(tokens)
# print(text)



        
