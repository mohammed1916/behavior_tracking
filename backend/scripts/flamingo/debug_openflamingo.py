#!/usr/bin/env python3
"""
Direct test of OpenFlamingo with comprehensive debugging
"""
import os
import sys
import signal
import types
import traceback
import transformers
import torch

# Set environment variables
os.environ['TRUST_REMOTE_CODE'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Fix Windows signal issue
if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14
    signal.alarm = lambda x: None

print("🔧 Starting OpenFlamingo debug test...")

# Comprehensive transformers patching
print("📦 Patching transformers...")
original_config_from_pretrained = transformers.AutoConfig.from_pretrained
original_tokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained
original_model_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

def patched_config_from_pretrained(*args, **kwargs):
    kwargs['trust_remote_code'] = True
    print(f"   Config loading: {args[0]} with trust_remote_code=True")
    return original_config_from_pretrained(*args, **kwargs)

def patched_tokenizer_from_pretrained(*args, **kwargs):
    kwargs['trust_remote_code'] = True
    print(f"   Tokenizer loading: {args[0]} with trust_remote_code=True")
    return original_tokenizer_from_pretrained(*args, **kwargs)

def patched_model_from_pretrained(*args, **kwargs):
    kwargs['trust_remote_code'] = True
    print(f"   Model loading: {args[0]} with trust_remote_code=True")
    model = original_model_from_pretrained(*args, **kwargs)
    
    # Comprehensive embedding method injection for MPT models
    print(f"   Model type: {type(model).__name__}")
    print(f"   Model class: {model.__class__}")
    
    # Check if it's an MPT model that needs embedding methods
    if 'MPT' in type(model).__name__ or 'mpt' in str(type(model)).lower():
        print(f"   🎯 Detected MPT model: {type(model).__name__}")
        
        if not hasattr(model, 'get_input_embeddings'):
            print("   🔧 Injecting get_input_embeddings...")
            def get_input_embeddings(self):
                if hasattr(self, 'transformer') and hasattr(self.transformer, 'wte'):
                    print(f"      Found transformer.wte: {type(self.transformer.wte)}")
                    return self.transformer.wte
                elif hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
                    print(f"      Found model.embed_tokens: {type(self.model.embed_tokens)}")
                    return self.model.embed_tokens
                else:
                    print("      No embedding layer found!")
                    return None
            
            model.get_input_embeddings = types.MethodType(get_input_embeddings, model)
        
        if not hasattr(model, 'set_input_embeddings'):
            print("   🔧 Injecting set_input_embeddings...")
            def set_input_embeddings(self, value):
                if hasattr(self, 'transformer') and hasattr(self.transformer, 'wte'):
                    print(f"      Setting transformer.wte to: {type(value)}")
                    self.transformer.wte = value
                elif hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
                    print(f"      Setting model.embed_tokens to: {type(value)}")
                    self.model.embed_tokens = value
                else:
                    print("      No embedding layer found to set!")
            
            model.set_input_embeddings = types.MethodType(set_input_embeddings, model)
        
        # Test the methods
        try:
            embeddings = model.get_input_embeddings()
            print(f"   ✅ get_input_embeddings() works: {type(embeddings)}")
        except Exception as e:
            print(f"   ❌ get_input_embeddings() failed: {e}")
        
        print("   ✅ MPT model patched successfully")
    
    return model

# Apply patches
transformers.AutoConfig.from_pretrained = patched_config_from_pretrained
transformers.AutoTokenizer.from_pretrained = patched_tokenizer_from_pretrained
transformers.AutoModelForCausalLM.from_pretrained = patched_model_from_pretrained

try:
    print("📥 Importing OpenFlamingo...")
    from open_flamingo import create_model_and_transforms
    
    print("🚀 Creating OpenFlamingo model...")
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
        decoder_layers_attr_name="transformer.blocks"
    )
    
    print("✅ SUCCESS! OpenFlamingo model created!")
    print(f"   Model: {type(model)}")
    print(f"   Image processor: {type(image_processor)}")
    print(f"   Tokenizer: {type(tokenizer)}")
    
    # Test with a simple image
    from PIL import Image
    import numpy as np
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    print("🖼️ Testing with simple image...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Process image
    vision_x = image_processor(test_image).unsqueeze(0).to(device)
    print(f"   Vision input shape: {vision_x.shape}")
    
    # Prepare text prompt
    text_prompt = "An image of"
    tokenized_text = tokenizer(text_prompt, return_tensors="pt").to(device)
    print(f"   Text input shape: {tokenized_text['input_ids'].shape}")
    
    print("🎯 Generating caption...")
    with torch.no_grad():
        generated_tokens = model.generate(
            vision_x=vision_x,
            lang_x=tokenized_text["input_ids"],
            attention_mask=tokenized_text["attention_mask"],
            max_new_tokens=10,
            do_sample=False,
        )
    
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    caption = generated_text.replace(text_prompt, "").strip()
    
    print(f"🎉 SUCCESS! Generated caption: '{caption}'")

except Exception as e:
    print(f"❌ OpenFlamingo test failed: {str(e)}")
    traceback.print_exc()

finally:
    # Restore original methods
    transformers.AutoConfig.from_pretrained = original_config_from_pretrained
    transformers.AutoTokenizer.from_pretrained = original_tokenizer_from_pretrained
    transformers.AutoModelForCausalLM.from_pretrained = original_model_from_pretrained
    print("🔄 Restored original transformers methods")

print("✨ OpenFlamingo debug test complete!")