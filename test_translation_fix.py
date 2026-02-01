#!/usr/bin/env python3
"""
Quick test to verify the translation pipeline fix
"""

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

print("Testing translation pipeline fix...")
print("This will download models on first run - it may take a few minutes")

# Using a smaller model for faster testing
model_name = "facebook/nllb-200-distilled-600M"
src_lang = "eng_Latn"
tgt_lang = "spa_Latn"

print(f"\nLoading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Creating translation pipeline...")
# The fix: Don't specify task as "translation" - let it infer from model
translator = pipeline(
    model=model,
    tokenizer=tokenizer,
    src_lang=src_lang,
    tgt_lang=tgt_lang,
    max_length=400,
    device=-1,  # CPU
)

print("Pipeline created successfully!")
print("\nTesting translation...")
test_text = "Hello, how are you?"
result = translator(test_text)
print(f"Input: {test_text}")
print(f"Output: {result[0]['translation_text']}")
print("\n✓ Translation pipeline is working correctly!")
