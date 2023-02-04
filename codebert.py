from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
with open("test.c", "r") as f:
    code = f.read()

code_tokens = tokenizer.tokenize(code)
print(code_tokens)
print("\n\n\n")
tokens=[tokenizer.cls_token]+[tokenizer.sep_token]+code_tokens+[tokenizer.eos_token]
tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]

print(context_embeddings.size())
np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)
for i in range(context_embeddings.shape[1]):
    print(np.matrix(context_embeddings[0][i].detach().numpy()))
