#!/usr/bin/env python3
"""
generator.py
Given a query and retrieved chunks, construct a prompt and call local LLaMA (gguf via llama-cpp-python)
OR fall back to HF transformers if GGUF not present.

Provides a simple function `answer_query(...)` that returns answer text and provenance.

Usage:
    from generator import LLMGenerator, build_prompt, answer_query
"""

import os
import json
from typing import List
from pathlib import Path
from transformers import GenerationConfig
from transformers import GenerationConfig
import torch

# try llama-cpp-python first (for gguf)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except Exception:
    LLAMA_CPP_AVAILABLE = False

# fallback to transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Default settings
DEFAULT_GGUF = "models/llama-3.2-3b-instruct.gguf"
DEFAULT_HF_FOLDER = "models/llama-3.2-3b-instruct"  # if you have HF safetensors folder
SYSTEM_INSTRUCTION = (
    "You are an assistant answering user questions using ONLY the provided excerpts. "
    "If the answer is not in the provided context, reply exactly: \"Not covered in the document.\" "
    "Cite chunk ids that you used in your answer."
)

DEFAULT_LLAMA_PARAMS = dict(temperature=0.0, top_p=0.95, max_tokens=512, n_ctx=2048)

class LLMGenerator:
    def __init__(self, gguf_path=DEFAULT_GGUF, hf_folder=DEFAULT_HF_FOLDER, llm_params=None):
        self.gguf_path = gguf_path
        self.hf_folder = hf_folder
        self.llm_params = llm_params or DEFAULT_LLAMA_PARAMS
        self.mode = None

        # prefer llama.cpp
        if LLAMA_CPP_AVAILABLE and Path(self.gguf_path).exists():
            self.mode = "gguf"
            self._init_gguf()
        elif TRANSFORMERS_AVAILABLE and Path(self.hf_folder).exists():
            self.mode = "hf"
            self._init_hf()
        else:
            raise RuntimeError("No usable LLM backend found. Put a GGUF at models/*.gguf or a HF folder at models/*.")

    def _init_gguf(self):
        # llama-cpp-python instantiation
        print("Initializing llama-cpp-python with", self.gguf_path)
        self.llm = Llama(model_path=self.gguf_path)

    def _init_hf(self):
        import torch
        print("Initializing transformers from", self.hf_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_folder, use_fast=True)

    # Load model with accelerate/device_map="auto" (we already used that earlier in __init__)
    # If you want to force a device map, change the call here, but be careful with accelerate.
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_folder, device_map="auto", torch_dtype="auto")

    # transformers pipeline should NOT be given a device argument when the model is already dispatched by accelerate.
    # Create the pipeline without device=... so it uses the model's current device placement.
        try:
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        # Informational: where the model parameters are located (useful for debug)
            device_type = getattr(self.model.device, "type", None) if hasattr(self.model, "device") else None
            print("Transformers pipeline created. model.device (approx):", device_type)
        except ValueError as e:
        # Fallback: often this happens when accelerate/device_map placement is unusual.
            print("Warning: could not create pipeline without device. Falling back to direct generate() usage.", str(e))
            self.pipeline = None

    def build_prompt(self, query: str, retrieved: List[dict]) -> str:
        # assemble context with chunk ids + short header
        ctx = ""
        for r in retrieved:
            ctx += f"[Chunk {r['chunk_id']} | score:{r['score']:.3f} | source:{r['source']}]\n{r['text']}\n\n"
        prompt = f"{SYSTEM_INSTRUCTION}\n\nContext:\n{ctx}\nQuestion: {query}\n\nAnswer (concise, cite chunk ids):"
        return prompt

    def generate_gguf(self, prompt: str):
        params = self.llm_params.copy()
        # llama-cpp-python uses different param names; create wrapper
        gen = self.llm.create(prompt=prompt,
                              temperature=float(params.get("temperature", 0.0)),
                              top_p=float(params.get("top_p", 0.95)),
                              max_tokens=int(params.get("max_tokens", 512)))
        txt = gen["choices"][0]["text"]
        return txt
    
    def generate_hf(self, prompt: str):
        params = self.llm_params

        gen_cfg = GenerationConfig(
            temperature=params.get("temperature", 0.0),
            top_p=params.get("top_p", 0.95),
            do_sample=(params.get("temperature", 0.0) > 0),
            max_new_tokens=params.get("max_tokens", 128),   # REDUCED!
            repetition_penalty=1.4,                         # FIX REPETITION!
            pad_token_id=self.tokenizer.eos_token_id
        )

    # FORCE the model to follow instructions:
        strict_prompt = (
            prompt
            + "\n\nYOUR RESPONSE MUST BE ONE OR TWO SENTENCES ONLY. "
              "DO NOT ELABORATE. DO NOT ADD NOTES. DO NOT REPEAT. "
              "DO NOT SAY ANYTHING OUTSIDE THE CONTEXT. KEEP PROPER SPACING.\n\n"
              "FINAL ANSWER:"
        )

        inputs = self.tokenizer(strict_prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_cfg,
                eos_token_id=self.tokenizer.eos_token_id
            ) 
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    
    def answer_query(self, query: str, retrieved: List[dict]):
        prompt = self.build_prompt(query, retrieved)
        if self.mode == "gguf":
            ans = self.generate_gguf(prompt)
        else:
            ans = self.generate_hf(prompt)
        return {"answer": ans.strip(), "prompt": prompt, "retrieved": retrieved}


if __name__ == "__main__":
    # quick CLI test that expects the Retriever to provide retrieval
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default=DEFAULT_GGUF)
    p.add_argument("--hf", default=DEFAULT_HF_FOLDER)
    args = p.parse_args()

    # minimal interactive test
    # Note: this will error if neither backend is available
    gen = LLMGenerator(gguf_path=args.gguf, hf_folder=args.hf)
    print("LLM backend:", gen.mode)
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            continue
        # For demo, ask user to paste a short retrieved context or type 'demo'
        print("Enter a short context (or type 'demo' to use placeholder):")
        ctx = input()
        if ctx.strip().lower() == "demo":
            retrieved = [{"chunk_id": 0, "score": 0.9, "source": "demo", "text": "This is a demo chunk that defines gradient descent..."}]
        else:
            retrieved = [{"chunk_id": 0, "score": 0.9, "source": "user", "text": ctx}]
        out = gen.answer_query(q, retrieved)
        print("\n=== ANSWER ===\n")
        print(out["answer"])