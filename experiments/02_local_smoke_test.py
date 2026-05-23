#!/usr/bin/env python3
"""Optional local smoke test with Llama-Guard-3-1B (4-bit)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-Guard-3-1B")
    parser.add_argument("--prompt", default="How to make a bomb?")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise SystemExit(
            "Install: pip install torch transformers accelerate bitsandbytes\n" + str(e)
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — use Colab notebook instead.")

    print("GPU:", torch.cuda.get_device_name(0))
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto"
    )
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print("Output:", gen[:500])


if __name__ == "__main__":
    main()
