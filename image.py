import sys
try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    import json
    import os
    import time
    from diffusers import DiffusionPipeline
    import sys
    import gc
    import contextlib

#  SETUP 
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    print(f"Device: {device.upper()}")
    print("Loading model...")
    
    
    pipe = None
    
    try:
        
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()
        
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            use_safetensors=True,  
            variant="fp16" if use_cuda else None,  
            local_files_only=False  
        )
        
        if pipe is None:
            raise RuntimeError("Pipeline returned None")
            
        print("Model loaded successfully")
        
        pipe.set_progress_bar_config(disable=True)
        
        if use_cuda:
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        else:
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
        
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
            
    except Exception as e:
        print(f"Model loading failed at stage: {e}")
        print("Cache location: ~/.cache/huggingface/diffusers/")
        raise

    if pipe is None:
        raise RuntimeError("Pipeline failed to load. Exiting.")

    # UTILITY FUNCTIONS
    def clean_prompt(p):
        """Clean and trim prompt"""
        p = " ".join(p.replace("\n", " ").split())
        words = p.split()
        return " ".join(words[:60])

    def enhance_prompt(p):
        return p + ", realistic face, natural pose"
    try:
        import psutil
        def get_cpu_temp():
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name in temps:
                        return temps[name][0].current
                return 0
            except:
                return 0
    except ImportError:
        print("psutil not installed, CPU temp monitoring disabled")
        def get_cpu_temp():
            return 0

    import subprocess
    def get_gpu_temp():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
            return 0
        except:
            return 0
    
    def wait_for_cooldown(max_temp=80, check_interval=5, max_wait=300):
        """Pause generation until temps drop, with timeout"""
        start_time = time.time()
        while True:
            cpu_temp = get_cpu_temp()
            gpu_temp = get_gpu_temp()
            
            if cpu_temp == 0 and gpu_temp == 0:
                break  
                
            if cpu_temp < max_temp and gpu_temp < (max_temp + 10):
                break
                
            if time.time() - start_time > max_wait:
                print(f"Cooldown timeout after {max_wait}s, continuing anyway...")
                break
                
            print(f"CPU: {cpu_temp}°C | GPU: {gpu_temp}°C - Waiting {check_interval}s...")
            time.sleep(check_interval)

    
    storyboard_file = r"D:/vs code/project/storyboard.json"
    if not os.path.exists(storyboard_file):
        raise FileNotFoundError(f"Storyboard file not found: {storyboard_file}")
        
    with open(storyboard_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data["scenes"]
   
    print(f"Loaded {len(scenes)} scenes from storyboard")
    
    #BATCHED IMAGE GENERATION
    BATCH_SIZE = 2
    INFERENCE_STEPS = 3
    RESOLUTION = 768

    NEGATIVE_PROMPT = """ cartoon, illustration, anime, painting, drawing, blurry, low quality, distorted, bad anatomy, duplicates, close up shots, duplicate hands, duplicate fingers, extra hands, extra fingers, duplicate humans, distorted papers, close-up, zoomed in, cropped face, portrait close-up, missing hands, missing arms, broken hands, no-tearing,
    deformed hands"""

    os.makedirs("outputs", exist_ok=True)

    print(f"Starting image generation on {device.upper()}...")

    for i in range(0, len(scenes), BATCH_SIZE):
        try:
            batch = scenes[i:i+BATCH_SIZE]

           
        
            STYLE = " sharp focus, high detail, detailed face, wide-shot, medium shot, upper body visible, symmetrical face"

            prompts = [
                clean_prompt(enhance_prompt(scene["visual_prompt"]) + ", " + STYLE)
            for scene in batch
        ]

            for idx, p in enumerate(prompts):
                print(f"\nScene {batch[idx]['scene_number']} Prompt:")
                print(p)

            
            with torch.no_grad():
                results = pipe(
                    prompts,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=RESOLUTION,
                    width=RESOLUTION,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=1.0
                ).images

            
            for idx, img in enumerate(results):

                scene = batch[idx]

                image_path = f"outputs/scene_{scene['scene_number']}.png"
                img.save(image_path)

                #Update JSON fields
                scene["image_path"] = image_path
                scene["status"] = "image_generated"

                print(f"Image saved: {image_path}")

            #Save updated JSON after every batch
            with open(storyboard_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print("JSON updated")
        except Exception as e:
            print(f"Error in scene {i}: {e}")
            continue  
        
        finally:
            torch.cuda.empty_cache()
            time.sleep(1)
            wait_for_cooldown(max_temp=75) #Remove if not in local

    
    print("\nAll images generated!")

except KeyboardInterrupt:
    print("\nExiting (User interruption)")

except Exception as e:
    print(f"\nUnexpected error: {e}")

finally:
    print("Final cleanup")
    try:
        torch.cuda.empty_cache()
    except:
        pass
    sys.exit(0)