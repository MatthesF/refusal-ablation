import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import MODEL_ID, MODEL_REVISION


def load_gemma():
    # Use the fastest local backend available on the machine running the experiment.
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def chat_tokens(tokenizer, prompt, device):
    # Gemma-instruct expects the user prompt wrapped in its chat template.
    text = prompt
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer(text, return_tensors="pt").to(device)


def last_prompt_token_activations(model, tokenizer, prompt, device):
    tokens = chat_tokens(tokenizer, prompt, device)

    # We use the final token of the prompt because this is where the model has read
    # the whole request but has not started generating an answer yet.
    prompt_last = int(tokens["attention_mask"].sum().item() - 1)

    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True, use_cache=False)

    # One activation vector per layer, taken at the final prompt token.
    return torch.stack([
        hidden[0, prompt_last, :].detach().float().cpu()
        for hidden in output.hidden_states[1:]
    ]).numpy()


def generate(model, tokenizer, prompt, device, max_new_tokens):
    tokens = chat_tokens(tokenizer, prompt, device)

    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = tokens["input_ids"].shape[1]
    return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()


def refusal_directions(activations, labels):
    labels = np.asarray(labels, dtype=str)

    # Each mean has shape [layer, hidden_dim].
    safe_mean = activations[labels == "safe"].mean(axis=0)
    unsafe_mean = activations[labels == "unsafe"].mean(axis=0)

    # Unsafe minus safe gives one candidate refusal direction per layer.
    direction = unsafe_mean - safe_mean

    # Biprojection: remove the part of that direction aligned with normal safe prompts.
    safe_axis = safe_mean / np.linalg.norm(safe_mean, axis=1, keepdims=True)
    direction = direction - np.sum(direction * safe_axis, axis=1, keepdims=True) * safe_axis

    # The edit only needs directions, so store unit vectors.
    return (direction / np.linalg.norm(direction, axis=1, keepdims=True)).astype(np.float32)


def edit_model(model, directions):
    with torch.no_grad():
        for layer, direction in zip(model.model.language_model.layers, directions):
            # Edit both main projections that write back into the residual stream.
            layer.self_attn.o_proj.weight.copy_(remove_direction(layer.self_attn.o_proj.weight, direction))
            layer.mlp.down_proj.weight.copy_(remove_direction(layer.mlp.down_proj.weight, direction))


def remove_direction(weight, direction):
    weight_float = weight.detach().float()
    direction = torch.as_tensor(direction, device=weight.device, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction)

    # Linear weights are [output, input], so each column is one output-space vector.
    old_norms = torch.linalg.vector_norm(weight_float, dim=0, keepdim=True)

    # Subtract each column's component in the refusal direction.
    projection = direction.unsqueeze(0) @ weight_float
    edited = weight_float - direction.unsqueeze(1) @ projection
    new_norms = torch.linalg.vector_norm(edited, dim=0, keepdim=True)

    # Remove the direction but keep each weight column at its original norm.
    scale = torch.where(old_norms == 0, torch.ones_like(old_norms), old_norms / new_norms)
    return (edited * scale).to(dtype=weight.dtype)
