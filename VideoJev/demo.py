import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor

video = hf_hub_download(
    repo_id="ZhuYuhan/VideoJev-Demo", filename="eating_spaghetti.mp4",
    repo_type="dataset", revision="7e2b9685bca592c3b929268f6fcf82c4ad74b267",
)
repo = "ZhuYuhan/VideoJev-4B"
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, attn_implementation="flash_attention_2",
)
model.prepare_for_inference()

questions = [
    {"id": "grounding-1", "task": "grounding",
     "query": "The woman in green eats pasta."},
    {"id": "qa-1", "task": "choice",
     "query": "What is the woman in green eating?",
     "options": ["A. Pasta.", "B. Rice.", "C. Soup."]},
]
results = model.predict_video(
    processor, video, questions,
    fps=2.0, max_frames=2048, context=8192,
)
print(json.dumps(results, indent=2))