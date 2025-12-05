#!/usr/bin/env python3
"""
Investigate MosaicGPTConfig attributes
"""
import os
os.environ['TRUST_REMOTE_CODE'] = 'True'

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    
    print("🔍 Investigating MosaicGPTConfig...")
    
    # Load the config
    config = AutoConfig.from_pretrained("anas-awadalla/mpt-1b-redpajama-200b", trust_remote_code=True)
    print(f"Config type: {type(config)}")
    
    print("\n📋 Available config attributes:")
    for attr in sorted(dir(config)):
        if not attr.startswith('_'):
            try:
                value = getattr(config, attr)
                if not callable(value):
                    print(f"   {attr}: {value}")
            except:
                print(f"   {attr}: <property>")
    
    # Look for hidden size equivalents
    print("\n🎯 Looking for hidden size equivalents...")
    potential_attrs = ['d_model', 'n_embd', 'hidden_size', 'dim', 'embedding_size']
    for attr in potential_attrs:
        if hasattr(config, attr):
            print(f"   ✅ Found {attr}: {getattr(config, attr)}")
        else:
            print(f"   ❌ Missing {attr}")
    
    # Load model to check actual dimensions
    print("\n🏗️ Loading model to check dimensions...")
    model = AutoModelForCausalLM.from_pretrained("anas-awadalla/mpt-1b-redpajama-200b", trust_remote_code=True)
    
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        embeddings = model.transformer.wte
        print(f"   Embedding dimensions: {embeddings.weight.shape}")
        hidden_dim = embeddings.weight.shape[1]
        print(f"   Hidden dimension: {hidden_dim}")
        
        # Add hidden_size to config if missing
        if not hasattr(config, 'hidden_size'):
            config.hidden_size = hidden_dim
            print(f"   ✅ Added hidden_size to config: {config.hidden_size}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✨ Investigation complete!")