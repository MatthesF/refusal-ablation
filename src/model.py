import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import MODEL_ID, MODEL_REVISION


def load_gemma():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

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
    text = prompt
    if tokenizer.chat_template:
        # Gemma-instruct expects the user prompt wrapped in its chat format.
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


def generate(model, tokenizer, prompt, device):
    tokens = chat_tokens(tokenizer, prompt, device)
    prompt_length = tokens["input_ids"].shape[1]
    max_new_tokens = int(model.config.max_position_embeddings) - prompt_length
    if max_new_tokens <= 0:
        raise ValueError("prompt is longer than the model context window")

    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_tokens = output[0, prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    token_ids = answer_tokens.tolist()
    ended_on_eos = tokenizer.eos_token_id in token_ids if tokenizer.eos_token_id is not None else False
    hit_token_limit = len(answer_tokens) >= max_new_tokens and not ended_on_eos
    return answer, len(answer_tokens), max_new_tokens, hit_token_limit


def refusal_directions(activations, labels):
    labels = np.asarray(labels, dtype=str)
    if set(labels) != {"safe", "unsafe"}:
        raise ValueError("refusal directions need both safe and unsafe labels")

    # Each mean has shape [layer, hidden_dim].
    safe_mean = activations[labels == "safe"].mean(axis=0)
    unsafe_mean = activations[labels == "unsafe"].mean(axis=0)

    # Unsafe minus safe gives one candidate refusal direction per layer.
    direction = unsafe_mean - safe_mean

    # Extra Gemma recipe step: orthogonalize the direction against the safe mean.
    safe_axis = safe_mean / np.linalg.norm(safe_mean, axis=1, keepdims=True)
    direction = direction - np.sum(direction * safe_axis, axis=1, keepdims=True) * safe_axis

    # The edit only needs directions, so store unit vectors.
    return (direction / np.linalg.norm(direction, axis=1, keepdims=True)).astype(np.float32)


def edit_model(model, directions):
    layers = model.model.language_model.layers
    if len(directions) != len(layers):
        raise ValueError("number of directions must match number of language layers")

    with torch.no_grad():
        for layer, direction in zip(layers, directions):
            attention_output = layer.self_attn.o_proj
            mlp_output = layer.mlp.down_proj

            # Edit both main projections that write back into the residual stream.
            attention_output.weight.copy_(remove_direction(attention_output.weight, direction))
            mlp_output.weight.copy_(remove_direction(mlp_output.weight, direction))


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
    degenerate = (old_norms == 0) | (new_norms == 0)
    scale = torch.where(degenerate, torch.ones_like(old_norms), old_norms / new_norms)
    return (edited * scale).to(dtype=weight.dtype)
