#!/usr/bin/env python3
"""Generate 98 portrait images (3.jpg - 100.jpg) using Replicate's z-image-turbo model."""

import os
import random
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
MODEL_URL = "https://api.replicate.com/v1/models/prunaai/z-image-turbo/predictions"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "Prefer": "wait",
}

# Diverse prompt components
genders = ["a man", "a woman"]
ages = ["young", "middle-aged", "elderly", "teenage"]
ethnicities = [
    "East Asian", "South Asian", "Black", "White", "Hispanic", "Middle Eastern",
    "Southeast Asian", "Pacific Islander", "Native American", "mixed-race",
]
hair_styles = [
    "short hair", "long hair", "curly hair", "straight hair", "wavy hair",
    "braided hair", "buzz cut", "bald", "afro", "ponytail",
]
clothing = [
    "wearing a t-shirt", "wearing a button-up shirt", "wearing a sweater",
    "wearing a jacket", "wearing a hoodie", "wearing a blouse",
    "wearing a suit", "wearing a dress", "wearing a polo shirt",
    "wearing a cardigan",
]
expressions = [
    "neutral expression", "smiling", "serious expression", "slight smile",
    "confident expression", "calm expression", "friendly expression",
    "thoughtful expression",
]
backgrounds = [
    "plain white background", "plain grey background", "soft gradient background",
    "studio lighting", "natural lighting", "warm lighting",
]


def make_prompt(index):
    random.seed(index + 42)
    gender = random.choice(genders)
    age = random.choice(ages)
    ethnicity = random.choice(ethnicities)
    hair = random.choice(hair_styles)
    cloth = random.choice(clothing)
    expr = random.choice(expressions)
    bg = random.choice(backgrounds)
    return (
        f"portrait photo of {age} {ethnicity} {gender} with {hair}, "
        f"{cloth}, {expr}, {bg}, high quality, photorealistic, headshot"
    )


def generate_image(index):
    prompt = make_prompt(index)
    payload = {
        "input": {
            "width": 640,
            "height": 832,
            "prompt": prompt,
            "go_fast": False,
            "output_format": "jpg",
            "guidance_scale": 0,
            "output_quality": 80,
            "num_inference_steps": 8,
        }
    }

    output_path = os.path.join(OUTPUT_DIR, f"{index}.jpg")

    try:
        resp = requests.post(MODEL_URL, json=payload, headers=HEADERS, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "succeeded" and data.get("output"):
            img_url = data["output"]
            img_resp = requests.get(img_url, timeout=60)
            img_resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(img_resp.content)
            print(f"[OK] {index}.jpg - {prompt[:60]}...")
            return index, True
        else:
            print(f"[FAIL] {index}.jpg - status: {data.get('status')}, error: {data.get('error')}")
            return index, False
    except Exception as e:
        print(f"[ERROR] {index}.jpg - {e}")
        return index, False


def main():
    indices = list(range(3, 101))  # 3.jpg through 100.jpg
    print(f"Generating {len(indices)} images...")

    succeeded = 0
    failed = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(generate_image, i): i for i in indices}
        for future in as_completed(futures):
            idx, ok = future.result()
            if ok:
                succeeded += 1
            else:
                failed.append(idx)
            if succeeded % 10 == 0 and succeeded > 0:
                print(f"  Progress: {succeeded}/{len(indices)} succeeded")

    print(f"\nDone! {succeeded}/{len(indices)} images generated successfully.")
    if failed:
        failed.sort()
        print(f"Failed indices: {failed}")


if __name__ == "__main__":
    main()
