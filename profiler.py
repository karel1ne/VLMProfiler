import os

DATA_DIR = "/data/mihejkoveg"
BASE_DIR = os.path.join(DATA_DIR, "VLMProfiler_runs")
os.makedirs(f"{DATA_DIR}/cache/huggingface", exist_ok=True)
os.makedirs(f"{DATA_DIR}/cache/datasets", exist_ok=True)
os.makedirs(f"{DATA_DIR}/cache/tmp", exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

os.environ["HF_HOME"] = f"{DATA_DIR}/cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = f"{DATA_DIR}/cache/datasets"
os.environ["TMPDIR"] = f"{DATA_DIR}/cache/tmp"

import torch
import time
import pynvml
import threading
import json
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
import evaluate
from PIL import Image

# CUDA_VISIBLE_DEVICES=3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_power_usage(handle):
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except pynvml.NVMLError:
        return 0.0

def run_profiling():
    pynvml.nvmlInit()
    physical_gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_idx)
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading model: {model_id} in 4-bit")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            quantization_config=quantization_config,
            device_map={"": 0}
        )
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return

    print("Loading datasets in streaming mode...")
    datasets_dict = {}
    sqa = load_dataset("derek-thomas/ScienceQA", split="validation", streaming=True)
    datasets_dict["ScienceQA"] = sqa.filter(lambda x: x['image'] is not None).take(3)

    tvqa = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
    datasets_dict["TextVQA"] = tvqa.take(3)

    coco = load_dataset("jxie/coco_captions", split="train", streaming=True)
    datasets_dict["COCO_Caption"] = coco.take(3)

    wer_metric = evaluate.load("wer", cache_dir=os.environ["HF_HOME"])
    results = {}

    prof_path = os.path.join(BASE_DIR, "torch_trace.json")
        
    def on_trace_ready(prof):
        prof.export_chrome_trace(prof_path)

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            skip_first=1,
            wait=1,
            warmup=1,
            active=3,
            repeat=1,
        ),
        record_shapes=True,
        with_stack=True,
        on_trace_ready=on_trace_ready
    )
    
    prof.start()
    for ds_name, ds in datasets_dict.items():
        print(f"Evaluating on {ds_name}...")
        ds_results = []
        
        for item in ds:
            if ds_name == "ScienceQA":
                image = item['image']
                question = item['question']
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                ground_truth = str(item['answer'])
            elif ds_name == "TextVQA":
                image = item['image']
                question = item['question']
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                ground_truth = item['answers'][0] if item.get('answers') else ""
            else:
                image = item['image']
                prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
                ground_truth = item['caption'] if 'caption' in item else ""

            if image is None:
                continue

            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = processor(text=prompt, images=image, return_tensors="pt").to(0)

            power_readings = []
            stop_event = threading.Event()
            
            def power_monitor():
                while not stop_event.is_set():
                    power_readings.append(get_power_usage(handle))
                    time.sleep(0.01)
                    
            monitor_thread = threading.Thread(target=power_monitor)
            
            torch.cuda.synchronize()
            start_time = time.time()
            monitor_thread.start()
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=32)
                
            torch.cuda.synchronize()
            end_time = time.time()
            stop_event.set()
            monitor_thread.join()
            
            latency = end_time - start_time
            avg_power = sum(power_readings) / len(power_readings) if power_readings else 0
            energy_joules = avg_power * latency
            
            generated_text = processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
            
            ds_results.append({
                "latency_s": latency,
                "energy_J": energy_joules,
                "avg_power_W": avg_power,
                "generated_text": generated_text,
                "ground_truth": ground_truth
            })
            
            prof.step()
            
        preds = [r["generated_text"] for r in ds_results]
        refs = [r["ground_truth"] for r in ds_results]
        try:
            ds_wer = wer_metric.compute(predictions=preds, references=refs)
        except Exception as e:
            ds_wer = None
            
        avg_latency = sum(r["latency_s"] for r in ds_results) / len(ds_results) if ds_results else 0
        avg_energy = sum(r["energy_J"] for r in ds_results) / len(ds_results) if ds_results else 0
        
        results[ds_name] = {
            "avg_latency_s": avg_latency,
            "avg_energy_J": avg_energy,
            "wer": ds_wer,
            "samples": ds_results
        }
    prof.stop()
            
    res_path = os.path.join(BASE_DIR, "llava_7b_results_final.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Profiling complete. Results saved to {res_path}")

if __name__ == "__main__":
    run_profiling()