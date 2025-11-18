import requests
from pprint import pprint
from transformers import AutoTokenizer
from math import exp

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    url = "http://localhost:30000/generate"
    
    payload = {
        "model": "llama3.1-8b",
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 1.0,
            "max_new_tokens": 0,
            "ignore_eos": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 5,
    }

    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    pprint(data)
    
    input_token_ids = []
    for i in range(len(data["meta_info"]["input_token_logprobs"])):
        input_token_ids.append(data["meta_info"]["input_token_logprobs"][i][1])
        
    for token_id in input_token_ids:
        token = tokenizer.decode([token_id])
        print(f"Token ID: {token_id}, {token}")
        
    full_logprobs = data["meta_info"]["input_top_logprobs"]
    
    print("Full distribution for input tokens:", full_logprobs)
    import ipdb; ipdb.set_trace()
    for i in range(1, len(full_logprobs)):
        logprob = [logprobs[0] for logprobs in full_logprobs[i]]
        prob = [exp(lp) for lp in logprob]
        print(f"Token Position Index: {i}, Probabilities Sum: {sum(prob)}")
