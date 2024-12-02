import requests
import json
import os
from transformers import AutoTokenizer
import transformers
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def GPT4(prompt,key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open('template.txt', 'r',encoding="utf-8") as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step, please reply in plain text and do not use any bold or bullet-point Markdown formatting."
    
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    # print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    response_txt = response.text
    # print(response_txt)
    obj=response.json()
    # print(obj)
    text=obj['choices'][0]['message']['content']
    # print(text)
    # print()
    return get_params_dict(text)

def get_params_dict(output_text):
    response = output_text
    # Find Final split ratio
    split_ratio_match = re.search(r"Final split ratio: (.*?)(?=\n|\Z)", response)
    if split_ratio_match:
        SR_hw_split_ratio = split_ratio_match.group(1)
        # print("Final split ratio:", final_split_ratio)
    else:
        SR_hw_split_ratio="NULL"
        # print("Final split ratio not found.")
    # Find Regioanl Prompt
    prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_match:
        SR_prompt = prompt_match.group(1).strip()
        # print("Regional Prompt:", regional_prompt)
    else:
        SR_prompt="NULL"
        # print("Regional Prompt not found.")

    HB_prompt_list_match = re.search(r"HB_prompt_list: (.*?)(?=\n|\Z)", response)
    if HB_prompt_list_match:
        HB_prompt_list = HB_prompt_list_match.group(1).strip()
        # print("sub_prompt_list:", sub_prompt_list)
    else:
        HB_prompt_list="NULL"
        # print("sub_prompt_list not found.")

    HB_m_offset_list_match = re.search(r"HB_m_offset_list: (.*?)(?=\n|\Z)", response)
    if HB_m_offset_list_match:
        HB_m_offset_list = HB_m_offset_list_match.group(1).strip()
        # print("x_offset_list:", x_offset_list)
    else:
        HB_m_offset_list="NULL"
        # print("x_offset_list not found.")
    
    HB_n_offset_list_match = re.search(r"HB_n_offset_list: (.*?)(?=\n|\Z)", response)
    if HB_n_offset_list_match:
        HB_n_offset_list = HB_n_offset_list_match.group(1).strip()
        # print("y_offset_list:", y_offset_list)
    else:
        HB_n_offset_list="NULL"
        # print("y_offset_list not found.")

    HB_m_scale_list_match = re.search(r"HB_m_scale_list: (.*?)(?=\n|\Z)", response)
    if HB_m_scale_list_match:
        HB_m_scale_list = HB_m_scale_list_match.group(1).strip()
        # print("x_scale_list:", x_scale_list)
    else:
        HB_m_scale_list="NULL"
        # print("x_scale_list not found.")

    HB_n_scale_list_match = re.search(r"HB_n_scale_list: (.*?)(?=\n|\Z)", response)
    if HB_n_scale_list_match:
        HB_n_scale_list = HB_n_scale_list_match.group(1).strip()
        # print("y_scale_list:", y_scale_list)
    else:
        HB_n_scale_list="NULL"
        # print("y_scale_list not found.")

    image_region_dict = {'SR_hw_split_ratio': SR_hw_split_ratio, 'SR_prompt': SR_prompt, 'HB_prompt_list': HB_prompt_list, 'HB_m_offset_list': HB_m_offset_list, 'HB_n_offset_list': HB_n_offset_list, 'HB_m_scale_list': HB_m_scale_list, 'HB_n_scale_list': HB_n_scale_list}
    return image_region_dict

def local_llm(prompt,model_path=None):
    if model_path==None:
        model_id = "Llama-2-13b-chat-hf" 
    else:
        model_id=model_path
    print('Using model:',model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
    with open('./data/RAG_template.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        print('waiting for LLM response')
        res = model.generate(**model_input, max_new_tokens=4096)[0]
        output=tokenizer.decode(res, skip_special_tokens=True)
        output = output.replace(textprompt,'')
    return get_params_dict(output)
    # return output