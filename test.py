import time
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Write me a letter to Sam Altman"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="casperhansen/vicuna-7b-v1.5-awq-smoothquant", **{'quantization': 'smoothquant'})

start = time.time()

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    tokens = output.outputs[0].token_ids
    end = time.time()
    elapsed = end-start
    
    print(output)
    print(len(tokens) / elapsed, 'tokens/s')