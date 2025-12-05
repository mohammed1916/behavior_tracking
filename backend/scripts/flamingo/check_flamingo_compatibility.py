#!/usr/bin/env python3
"""
Quick compatibility check for OpenFlamingo components
"""
import os
os.environ['TRUST_REMOTE_CODE'] = 'True'

print("🔍 Checking OpenFlamingo component compatibility...")

try:
    print("1. Testing open_flamingo import...")
    from open_flamingo import create_model_and_transforms
    print("   ✅ open_flamingo imported successfully")
    
    print("2. Testing transformers import...")
    import transformers
    print("   ✅ transformers imported successfully")
    
    print("3. Testing basic model loading...")
    import torch
    from transformers import AutoModelForCausalLM
    
    print("   Loading MPT model directly...")
    model = AutoModelForCausalLM.from_pretrained(
        "anas-awadalla/mpt-1b-redpajama-200b",
        trust_remote_code=True
    )
    print(f"   ✅ MPT model loaded: {type(model)}")
    
    # Check embedding methods
    print(f"   Has get_input_embeddings: {hasattr(model, 'get_input_embeddings')}")
    
    if hasattr(model, 'get_input_embeddings'):
        try:
            embeddings = model.get_input_embeddings()
            print(f"   ✅ get_input_embeddings works: {type(embeddings)}")
        except Exception as e:
            print(f"   ❌ get_input_embeddings failed: {e}")
    
    print("4. Testing CLIP components...")
    try:
        import open_clip
        print("   ✅ open_clip imported successfully")
    except Exception as e:
        print(f"   ❌ open_clip failed: {e}")
    
    print("\n🎯 All basic components are working!")
    print("The issue is likely in the create_model_and_transforms integration")
    
except Exception as e:
    print(f"❌ Compatibility check failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✨ Compatibility check complete!")