import requests
from pprint import pprint

if __name__ == "__main__":
    url = "http://localhost:30000/generate"

    payload = {
    "text": "The capital of France is",
    "sampling_params": {
        "temperature": 1.0,
        "max_new_tokens": 0,
    },
    "return_logprob": True,
    "logprob_start_len": 0,
}

    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    pprint(data)
