import sys
import traceback
import os

print('Starting simplified OpenFlamingo test...')

# Set environment variables
os.environ['HF_ALLOW_CODE_EVAL'] = '1'
os.environ['TRUST_REMOTE_CODE'] = 'True'

try:
    # First test just the MPT model loading
    print('Testing MPT model loading...')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Test MPT model with path from official OpenFlamingo documentation
    lang_encoder_path = "anas-awadalla/mpt-1b-redpajama-200b"
    print(f'Loading MPT tokenizer from: {lang_encoder_path}')
    
    tokenizer = AutoTokenizer.from_pretrained(lang_encoder_path, trust_remote_code=True)
    print('✓ MPT Tokenizer loaded successfully')
    
    print('Loading MPT model...')
    model = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    print('✓ MPT Model loaded successfully')
    
    # Test text generation
    print('Testing text generation...')
    input_text = "The weather today is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    model = model.to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 5, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Generated: "{generated_text}"')
    
    print('✓ Basic MPT functionality works!')
    
    # Now test OpenFlamingo create_model_and_transforms with proper error handling
    print('\nTesting OpenFlamingo integration...')
    try:
        from open_flamingo import create_model_and_transforms
        
        # First try: attempt direct creation (will likely fail due to trust_remote_code)
        print('Attempting basic OpenFlamingo model creation...')
        try:
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",  # Official path
                tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",     # Official path
                cross_attn_every_n_layers=1  # Official example uses 1
            )
            print('✓ OpenFlamingo model created successfully!')
            
        except Exception as e:
            print(f'✗ Direct OpenFlamingo creation failed: {e}')
            
            # Second try: Pre-load the tokenizer/model with trust_remote_code first
            if 'trust_remote_code' in str(e):
                print('Attempting workaround by pre-loading with trust_remote_code...')
                try:
                    # Pre-load the problematic components
                    from transformers import AutoTokenizer, AutoConfig
                    
                    lang_path = "anas-awadalla/mpt-1b-redpajama-200b"
                    print(f'Pre-loading tokenizer and config for {lang_path}...')
                    
                    # Load tokenizer and config with trust_remote_code
                    tokenizer_preload = AutoTokenizer.from_pretrained(lang_path, trust_remote_code=True)
                    config_preload = AutoConfig.from_pretrained(lang_path, trust_remote_code=True)
                    
                    print('✓ Pre-loading successful, now attempting OpenFlamingo creation...')
                    
                    # Now try OpenFlamingo again (may still fail but worth testing)
                    model, image_processor, tokenizer = create_model_and_transforms(
                        clip_vision_encoder_path="ViT-L-14",
                        clip_vision_encoder_pretrained="openai",
                        lang_encoder_path=lang_path,
                        tokenizer_path=lang_path,
                        cross_attn_every_n_layers=1
                    )
                    print('✓ OpenFlamingo model created successfully after pre-loading!')
                    
                except Exception as e2:
                    print(f'✗ Workaround also failed: {e2}')
                    
            else:
                print('This is a different issue, not trust_remote_code related')
    
    except ImportError as e:
        print(f'✗ OpenFlamingo library not available: {e}')
        
    print('\n--- Test Summary ---')
    print('✓ MPT model works standalone')
    if 'model' in locals() and 'image_processor' in locals():
        print('✓ OpenFlamingo integration successful!')
    else:
        print('✗ OpenFlamingo integration has issues')
        print('Recommendation: Use BLIP for now until OpenFlamingo compatibility is fixed')
    
except Exception as e:
    print('Error during basic model test:')
    traceback.print_exc()
    sys.exit(1)