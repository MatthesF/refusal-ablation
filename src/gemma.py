import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import GENERATION_MAX_NEW_TOKENS, GEMMA_MODEL_ID, GEMMA_MODEL_REVISION


def load_gemma():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        GEMMA_MODEL_ID,
        revision=GEMMA_MODEL_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        revision=GEMMA_MODEL_REVISION,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def chat_tokens(tokenizer, prompt, device):
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
    prompt_last = int(tokens["attention_mask"].sum().item() - 1)

    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True, use_cache=False)

    # Measure the state just before the model starts answering.
    return torch.stack([
        hidden[0, prompt_last, :].detach().float().cpu()
        for hidden in output.hidden_states[1:]
    ]).numpy()


def generate(model, tokenizer, prompt, device):
    if tokenizer.eos_token_id is None:
        raise ValueError("Gemma tokenizer must define eos_token_id for deterministic generation")
    if tokenizer.pad_token_id is None:
        raise ValueError("Gemma tokenizer must define pad_token_id for generation")

    tokens = chat_tokens(tokenizer, prompt, device)
    prompt_length = tokens["input_ids"].shape[1]
    max_new_tokens = generation_room(model, prompt_length)

    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_tokens = output[0, prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    token_ids = answer_tokens.tolist()
    ended_on_eos = tokenizer.eos_token_id in token_ids
    hit_token_limit = len(answer_tokens) >= max_new_tokens and not ended_on_eos
    return answer, len(answer_tokens), max_new_tokens, hit_token_limit


def generation_room(model, prompt_length):
    context_window = int(model.config.get_text_config().max_position_embeddings)
    available = int(context_window) - prompt_length
    if available <= 0:
        raise ValueError("prompt is longer than the model context window")
    return min(GENERATION_MAX_NEW_TOKENS, available)


def refusal_directions(activations, labels):
    labels = np.asarray(labels, dtype=str)
    if set(labels) != {"safe", "unsafe"}:
        raise ValueError("refusal directions need both safe and unsafe labels")

    safe_mean = activations[labels == "safe"].mean(axis=0)
    unsafe_mean = activations[labels == "unsafe"].mean(axis=0)
    direction = unsafe_mean - safe_mean

    # Remove the component that only points along the safe-prompt mean.
    safe_axis = normalize_rows(safe_mean, "safe activation mean")
    direction = direction - np.sum(direction * safe_axis, axis=1, keepdims=True) * safe_axis
    return normalize_rows(direction, "refusal direction").astype(np.float32)


def normalize_rows(values, name):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] == 0)
    if len(zero_rows):
        layers = ", ".join(str(layer) for layer in zero_rows.tolist())
        raise ValueError(f"{name} contains zero vectors for layer(s): {layers}")
    return values / norms


def edit_gemma(model, directions):
    layers = model.model.language_model.layers
    if len(directions) != len(layers):
        raise ValueError("number of directions must match number of language layers")

    with torch.no_grad():
        for layer, direction in zip(layers, directions):
            layer.self_attn.o_proj.weight.copy_(remove_direction(layer.self_attn.o_proj.weight, direction))
            layer.mlp.down_proj.weight.copy_(remove_direction(layer.mlp.down_proj.weight, direction))


def remove_direction(weight, direction):
    weight_float = weight.detach().float()
    direction = torch.as_tensor(direction, device=weight.device, dtype=torch.float32)
    if direction.numel() != weight_float.shape[0]:
        raise ValueError(
            "refusal direction width must match the Linear output dimension: "
            f"{direction.numel()} != {weight_float.shape[0]}"
        )
    direction_norm = torch.linalg.vector_norm(direction)
    if direction_norm.item() == 0:
        raise ValueError("refusal direction must be non-zero")
    direction = direction / direction_norm

    # Each Linear column contributes an output vector. Keep the original column
    # norm only when an orthogonal component remains after removing the direction.
    old_norms = torch.linalg.vector_norm(weight_float, dim=0, keepdim=True)
    projection = direction.unsqueeze(0) @ weight_float
    edited = weight_float - direction.unsqueeze(1) @ projection
    new_norms = torch.linalg.vector_norm(edited, dim=0, keepdim=True)

    collapsed = (old_norms != 0) & (new_norms == 0)
    if torch.any(collapsed):
        columns = torch.nonzero(collapsed.squeeze(0), as_tuple=False).flatten().cpu().tolist()
        preview = ", ".join(str(column) for column in columns[:10])
        raise ValueError(f"cannot preserve norm after removing direction for column(s): {preview}")

    zero_columns = old_norms == 0
    safe_new_norms = torch.where(zero_columns, torch.ones_like(new_norms), new_norms)
    scale = torch.where(zero_columns, torch.ones_like(old_norms), old_norms / safe_new_norms)
    return (edited * scale).to(dtype=weight.dtype)
