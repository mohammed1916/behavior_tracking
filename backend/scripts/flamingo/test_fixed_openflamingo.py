#!/usr/bin/env python3
"""
Fixed OpenFlamingo test with proper MosaicGPT class patching
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

print("🔧 Starting fixed OpenFlamingo test...")

# Advanced MosaicGPT class patching
def patch_mosaicgpt_class():
    """Patch the MosaicGPT class directly before model creation"""
    try:
        # Import transformers first to ensure modules are loaded
        import transformers
        
        # Find and patch all MosaicGPT classes in loaded modules
        for module_name, module in sys.modules.items():
            if module is None:
                continue
            
            # Look for MosaicGPT class
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        'MosaicGPT' in attr.__name__ and 
                        not hasattr(attr, '_patched_embeddings')):
                        
                        print(f"🎯 Patching {attr.__name__} in {module_name}")
                        
                        # Add embedding methods directly to the class
                        def get_input_embeddings(self):
                            if hasattr(self, 'transformer') and hasattr(self.transformer, 'wte'):
                                return self.transformer.wte
                            else:
                                # Fallback - search for embedding layer
                                for name, layer in self.named_modules():
                                    if 'embed' in name.lower() and hasattr(layer, 'weight'):
                                        return layer
                                return None
                        
                        def set_input_embeddings(self, value):
                            if hasattr(self, 'transformer') and hasattr(self.transformer, 'wte'):
                                self.transformer.wte = value
                            else:
                                print("Warning: Could not set input embeddings")
                        
                        # Patch the class
                        attr.get_input_embeddings = get_input_embeddings
                        attr.set_input_embeddings = set_input_embeddings
                        attr._patched_embeddings = True
                        
                        print(f"✅ Successfully patched {attr.__name__}")
                        return True
                        
                except Exception as e:
                    continue
                    
        return False
    except Exception as e:
        print(f"⚠️ Class patching error: {e}")
        return False

# Hook into model loading to patch after import but before instantiation
original_model_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

def patched_model_from_pretrained(*args, **kwargs):
    kwargs['trust_remote_code'] = True
    print(f"   Model loading: {args[0]} with trust_remote_code=True")
    
    # Patch MosaicGPT classes before creating the model
    patch_mosaicgpt_class()
    
    model = original_model_from_pretrained(*args, **kwargs)
    print(f"   Model created: {type(model).__name__}")
    
    return model

# Apply patch
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
    print(f"   Language model: {type(model.lang_encoder)}")
    
    # Test embedding methods
    try:
        embeddings = model.lang_encoder.get_input_embeddings()
        print(f"✅ get_input_embeddings() works: {type(embeddings)}")
    except Exception as e:
        print(f"❌ get_input_embeddings() failed: {e}")
    
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
    
    # Prepare text prompt
    text_prompt = "An image of"
    tokenized_text = tokenizer(text_prompt, return_tensors="pt").to(device)
    
    print("🎯 Generating caption...")
    with torch.no_grad():
        generated_tokens = model.generate(
            vision_x=vision_x,
            lang_x=tokenized_text["input_ids"],
            attention_mask=tokenized_text["attention_mask"],
            max_new_tokens=5,
            do_sample=False,
        )
    
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    caption = generated_text.replace(text_prompt, "").strip()
    
    print(f"🎉 SUCCESS! Generated caption: '{caption}'")

except Exception as e:
    print(f"❌ OpenFlamingo test failed: {str(e)}")
    traceback.print_exc()

finally:
    # Restore original method
    transformers.AutoModelForCausalLM.from_pretrained = original_model_from_pretrained
    print("🔄 Restored original transformers methods")

print("✨ Fixed OpenFlamingo test complete!")