#!/usr/bin/env python3
"""
Complete OpenFlamingo fix with MosaicGPT config and embedding patches
"""
import os
import signal
import torch
import transformers

# Set environment
os.environ['TRUST_REMOTE_CODE'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14

print("🚀 Testing OpenFlamingo with complete fix")
print("=" * 50)

def patch_mosaicgpt_config_and_model():
    """Comprehensive patch for MosaicGPT compatibility"""
    original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
    
    def patched_from_pretrained(*args, **kwargs):
        kwargs['trust_remote_code'] = True
        print(f"   Loading model: {args[0]}")
        
        model = original_from_pretrained(*args, **kwargs)
        
        # Fix 1: Add hidden_size to config if missing (uses d_model)
        if hasattr(model, 'config') and hasattr(model.config, 'd_model') and not hasattr(model.config, 'hidden_size'):
            model.config.hidden_size = model.config.d_model
            print(f"   ✅ Added hidden_size to config: {model.config.hidden_size}")
        
        # Fix 2: Patch MosaicGPT embedding methods
        if 'MosaicGPT' in str(type(model)):
            def get_input_embeddings(self):
                return self.transformer.wte
            
            def set_input_embeddings(self, value):
                self.transformer.wte = value
            
            model.__class__.get_input_embeddings = get_input_embeddings
            model.__class__.set_input_embeddings = set_input_embeddings
            print(f"   ✅ Patched MosaicGPT embedding methods")
        
        return model
    
    return patched_from_pretrained, original_from_pretrained

def test_openflamingo():
    try:
        print("1️⃣ Importing dependencies...")
        from open_flamingo import create_model_and_transforms
        from PIL import Image
        import numpy as np
        
        print("2️⃣ Setting up patches...")
        patched_method, original_method = patch_mosaicgpt_config_and_model()
        transformers.AutoModelForCausalLM.from_pretrained = patched_method
        
        try:
            print("3️⃣ Creating OpenFlamingo model...")
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
                tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
                cross_attn_every_n_layers=1,
                decoder_layers_attr_name="transformer.blocks"
            )
            print("   ✅ OpenFlamingo model created successfully!")
            
            print("4️⃣ Testing with simple image...")
            
            # Create test image
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            # Setup device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Using device: {device}")
            model.to(device)
            
            # Process image
            vision_x = image_processor(test_image).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            print(f"   Vision shape: {vision_x.shape}")
            
            # Prepare text
            tokenizer.padding_side = "left"
            text_prompt = "An image of"
            lang_x = tokenizer([f"<image>{text_prompt}"], return_tensors="pt")
            lang_x = {k: v.to(device) for k, v in lang_x.items()}
            print(f"   Text shape: {lang_x['input_ids'].shape}")
            
            print("5️⃣ Generating caption...")
            with torch.no_grad():
                generated_tokens = model.generate(
                    vision_x=vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    max_new_tokens=5,
                    do_sample=False,
                )
            
            # Decode result
            generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            caption = generated_text.replace(f"<image>{text_prompt}", "").strip()
            
            print(f"🎉 SUCCESS! Generated caption: '{caption}'")
            return True
            
        finally:
            # Restore original method
            transformers.AutoModelForCausalLM.from_pretrained = original_method
            print("   🔄 Restored original transformers methods")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_openflamingo()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 COMPLETE SUCCESS! OpenFlamingo is working!")
    else:
        print("💥 Test failed - see errors above")
    print("=" * 50)