#!/usr/bin/env python
import os
import torch
import logging

logging.basicConfig(level=logging.INFO)

# Set environment variables
os.environ['TRUST_REMOTE_CODE'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Testing OpenFlamingo with monkey-patching approach...")

try:
    from open_flamingo import create_model_and_transforms
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Store original methods
    original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained
    original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
    
    print("Applying monkey patches...")
    
    def patched_tokenizer_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        print(f"Patched tokenizer loading: {pretrained_model_name_or_path}")
        kwargs['trust_remote_code'] = True
        return original_tokenizer_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
    def patched_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        print(f"Patched model loading: {pretrained_model_name_or_path}")
        kwargs['trust_remote_code'] = True
        model = original_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # If this is an MPT model, patch the missing embedding methods
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            if not hasattr(model, 'get_input_embeddings'):
                def get_input_embeddings(self):
                    return self.transformer.wte
                def set_input_embeddings(self, value):
                    self.transformer.wte = value
                
                # Bind methods to the specific instance
                import types
                model.get_input_embeddings = types.MethodType(get_input_embeddings, model)
                model.set_input_embeddings = types.MethodType(set_input_embeddings, model)
                print("✓ Patched MPT model with embedding methods")
        
        return model
    
    # Apply monkey patches
    AutoTokenizer.from_pretrained = staticmethod(patched_tokenizer_from_pretrained)
    AutoModelForCausalLM.from_pretrained = staticmethod(patched_model_from_pretrained)
    
    print("Creating OpenFlamingo model...")
    
    # Pre-load and patch the MPT model directly
    print("Pre-loading MPT model to patch it...")
    mpt_model_path = "anas-awadalla/mpt-1b-redpajama-200b"
    
    # Load the model with trust_remote_code
    mpt_model = original_model_from_pretrained(mpt_model_path, trust_remote_code=True)
    
    # Patch the model with missing methods
    if not hasattr(mpt_model, 'get_input_embeddings'):
        def get_input_embeddings(self):
            return self.transformer.wte
        def set_input_embeddings(self, value):
            self.transformer.wte = value
        
        import types
        mpt_model.get_input_embeddings = types.MethodType(get_input_embeddings, mpt_model)
        mpt_model.set_input_embeddings = types.MethodType(set_input_embeddings, mpt_model)
        print("✓ Patched MPT model with embedding methods")
    
    # Now create a custom patched model function that returns our pre-loaded model
    def patched_model_return_preloaded(pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path == mpt_model_path:
            print(f"Returning pre-loaded and patched MPT model")
            return mpt_model
        else:
            kwargs['trust_remote_code'] = True
            return original_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    
    # Update the monkey patch to use our pre-loaded model
    AutoModelForCausalLM.from_pretrained = staticmethod(patched_model_return_preloaded)
    
    try:
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
            decoder_layers_attr_name="transformer.blocks"  # MPT model uses this attribute
        )
        
        print("✅ Success! OpenFlamingo model created with monkey-patching!")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Moving model to {device}...")
        model.to(device)
        model.eval()
        
        print("Testing basic functionality...")
        from PIL import Image
        test_img = Image.new('RGB', (224, 224), color='red')
        
        # Prepare the image following official documentation format
        vision_x = [image_processor(test_img).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)
        
        # Create a prompt for image captioning
        prompt_text = "<image>An image of"
        tokenizer.padding_side = "left"
        
        lang_x = tokenizer([prompt_text], return_tensors="pt")
        lang_x = {k: v.to(device) for k, v in lang_x.items()}
        
        # Generate caption
        with torch.no_grad():
            generated_text = model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=10,
                num_beams=1,
                do_sample=False
            )
        
        # Decode the generated text
        caption = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        caption = caption.replace(prompt_text, "").strip()
        
        print(f"✅ Generated caption: '{caption}'")
        print("OpenFlamingo is working with monkey-patching!")
        
    except Exception as e:
        print(f"❌ OpenFlamingo creation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original methods
        AutoTokenizer.from_pretrained = original_tokenizer_from_pretrained
        AutoModelForCausalLM.from_pretrained = original_model_from_pretrained
        print("Restored original methods")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()