import requests
from pprint import pprint
from transformers import AutoTokenizer
from math import exp

def check_if_all_index_included(full_logprobs, vocab_size):
    index_list = [logprob[1] for logprob in full_logprobs]
    
    return_value = True
    for i in range(vocab_size):
        if i not in index_list:
            print(f"Missing index: {i}")
            return_value = False
    
    return return_value

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    vocab_size = tokenizer.vocab_size
    url = "http://localhost:30000/generate"
    
    payload = {
        "model": "llama3.1-8b",
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0.75,
            "max_new_tokens": 0,
            "ignore_eos": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": vocab_size,
    }

    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # pprint(data)
    
    input_token_ids = []
    for i in range(len(data["meta_info"]["input_token_logprobs"])):
        input_token_ids.append(data["meta_info"]["input_token_logprobs"][i][1])
        
    for token_id in input_token_ids:
        token = tokenizer.decode([token_id])
        print(f"Token ID: {token_id}, {token}")
        
    full_logprobs = data["meta_info"]["input_top_logprobs"]
    
    # print("Full distribution for input tokens:", full_logprobs)

    for i in range(1, len(full_logprobs)):
        logprob = [logprobs[0] for logprobs in full_logprobs[i]]
        prob = [exp(lp) for lp in logprob]
        print(f"Token Position Index: {i}, Probabilities Sum: {sum(prob)}")
        
        all_included = check_if_all_index_included(full_logprobs[i], vocab_size)
        print(f"All indices included: {all_included}")