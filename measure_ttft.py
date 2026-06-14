import time
import requests

text = "a-na ša-lim-a-šùr qí-bi-ma"
times = []

for i in range(5):
    start = time.time()
    first_token = False
    
    response = requests.post(
        "http://localhost:8000/translate",
        json={"text": text},
        stream=True
    )
    
    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            data = line[6:].decode()
            if data != "[DONE]" and not first_token:
                ttft = time.time() - start
                times.append(ttft)
                first_token = True
                break
    
    print(f"Run {i+1}: TTFT = {ttft:.2f}s")

print(f"\nMedian TTFT: {sorted(times)[2]:.2f}s")
print(f"P95 TTFT: {max(times):.2f}s")