#!/usr/bin/env python3
"""
Simple test to load MosaicGPT and patch it directly
"""
import os
import sys
import signal

# Set environment variables
os.environ['TRUST_REMOTE_CODE'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Fix Windows signal issue
if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14

print("🔧 Testing direct MosaicGPT loading and patching...")

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("📥 Loading MosaicGPT model...")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "anas-awadalla/mpt-1b-redpajama-200b",
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded: {type(model)}")
    print(f"   Class: {model.__class__}")
    print(f"   Module: {model.__class__.__module__}")
    
    # Check current state
    print(f"   Has get_input_embeddings: {hasattr(model, 'get_input_embeddings')}")
    print(f"   Has transformer: {hasattr(model, 'transformer')}")
    
    if hasattr(model, 'transformer'):
        print(f"   Has transformer.wte: {hasattr(model.transformer, 'wte')}")
        if hasattr(model.transformer, 'wte'):
            print(f"   transformer.wte type: {type(model.transformer.wte)}")
    
    # Patch the instance directly
    print("🔧 Patching model instance...")
    
    def get_input_embeddings(self):
        return self.transformer.wte
    
    def set_input_embeddings(self, value):
        self.transformer.wte = value
    
    import types
    model.get_input_embeddings = types.MethodType(get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(set_input_embeddings, model)
    
    # Test the patched methods
    print("✅ Testing patched methods...")
    embeddings = model.get_input_embeddings()
    print(f"   get_input_embeddings() works: {type(embeddings)}")
    print(f"   Embeddings shape: {embeddings.weight.shape}")
    
    print("🎉 SUCCESS! MosaicGPT can be patched successfully!")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("✨ Direct MosaicGPT test complete!")