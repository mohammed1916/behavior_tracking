#!/usr/bin/env python3
"""
Standalone OpenFlamingo test with comprehensive error handling
"""
import os
import sys
import signal
import traceback

# Set environment for compatibility
os.environ['TRUST_REMOTE_CODE'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14

print("🚀 OpenFlamingo Standalone Test")
print("=" * 50)

def test_openflamingo():
    try:
        print("1️⃣ Importing dependencies...")
        import torch
        import transformers
        from open_flamingo import create_model_and_transforms
        from PIL import Image
        import numpy as np
        
        print("2️⃣ Setting up MosaicGPT patch...")
        
        # Store original method
        original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
        
        def patched_from_pretrained(*args, **kwargs):
            kwargs['trust_remote_code'] = True
            print(f"   Loading model: {args[0]}")
            
            # Patch MosaicGPT classes after import
            model = original_from_pretrained(*args, **kwargs)
            
            # Check if this is a MosaicGPT model and patch it
            if 'MosaicGPT' in str(type(model)):
                print(f"   Found MosaicGPT model: {type(model).__name__}")
                
                # Replace the problematic methods
                def get_input_embeddings(self):
                    return self.transformer.wte
                
                def set_input_embeddings(self, value):
                    self.transformer.wte = value
                
                # Patch the class, not instance
                model.__class__.get_input_embeddings = get_input_embeddings
                model.__class__.set_input_embeddings = set_input_embeddings
                print("   ✅ Patched MosaicGPT embedding methods")
            
            return model
        
        # Apply patch
        transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained
        
        print("3️⃣ Creating OpenFlamingo model...")
        try:
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
                tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
                cross_attn_every_n_layers=1,
                decoder_layers_attr_name="transformer.blocks"
            )
            print("   ✅ Model created successfully!")
            
        finally:
            # Restore original method
            transformers.AutoModelForCausalLM.from_pretrained = original_from_pretrained
        
        print("4️⃣ Testing with simple image...")
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        model.to(device)
        
        # Process image
        vision_x = image_processor(test_image).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        
        # Prepare text
        tokenizer.padding_side = "left"
        text_prompt = "An image of"
        lang_x = tokenizer([f"<image>{text_prompt}"], return_tensors="pt")
        lang_x = {k: v.to(device) for k, v in lang_x.items()}
        
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
        
        print(f"✅ SUCCESS! Generated caption: '{caption}'")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_openflamingo()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 OpenFlamingo test completed successfully!")
    else:
        print("💥 OpenFlamingo test failed - see errors above")
    print("=" * 50)