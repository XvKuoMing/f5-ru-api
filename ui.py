#!/usr/bin/env python3
import os
import gc
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    tempfile_kwargs,
)
from f5_tts.model import DiT

hf_token = os.environ.get("HF_TOKEN", None)

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
MODEL_REPO = "ESpeech/ESpeech-TTS-1_RL-V2"
MODEL_FILE = "espeech_tts_rlv2.pt"
VOCAB_FILE = "vocab.txt"

loaded_model = None

def ensure_model():
    global loaded_model
    if loaded_model is not None:
        return loaded_model
    model_path = None
    vocab_path = None
    print(f"Trying to download model file '{MODEL_FILE}' and '{VOCAB_FILE}' from hub '{MODEL_REPO}'")
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        vocab_path = hf_hub_download(repo_id=MODEL_REPO, filename=VOCAB_FILE)
        print(f"Downloaded model to {model_path}")
        print(f"Downloaded vocab to {vocab_path}")
    except Exception as e:
        print("hf_hub_download failed:", e)
    if model_path is None or vocab_path is None:
        try:
            local_dir = f"cache_{MODEL_REPO.replace('/', '_')}"
            print(f"Attempting snapshot_download into {local_dir}...")
            snapshot_dir = snapshot_download(repo_id=MODEL_REPO, cache_dir=None, local_dir=local_dir, token=hf_token)
            possible_model = os.path.join(snapshot_dir, MODEL_FILE)
            possible_vocab = os.path.join(snapshot_dir, VOCAB_FILE)
            if os.path.exists(possible_model):
                model_path = possible_model
            if os.path.exists(possible_vocab):
                vocab_path = possible_vocab
            print(f"Snapshot downloaded to {snapshot_dir}")
        except Exception as e:
            print("snapshot_download failed:", e)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found after download attempts: {model_path}")
    if not vocab_path or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found after download attempts: {vocab_path}")
    print(f"Loading model from: {model_path}")
    loaded_model = load_model(DiT, MODEL_CFG, model_path, vocab_file=vocab_path)
    return loaded_model

print("Preloading model...")
try:
    ensure_model()
    print("Model preloaded.")
except Exception as e:
    print(f"Model preload failed: {e}")

print("Loading RUAccent...")
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)
print("RUAccent loaded.")

print("Loading vocoder...")
vocoder = load_vocoder()
print("Vocoder loaded.")

def process_text_with_accent(text, accentizer):
    if not text or not text.strip():
        return text
    if '+' in text:
        return text
    else:
        return accentizer.process_all(text)

def process_texts_only(ref_text, gen_text):
    processed_ref_text = process_text_with_accent(ref_text, accentizer)
    processed_gen_text = process_text_with_accent(gen_text, accentizer)
    return processed_ref_text, processed_gen_text

def synthesize(
    ref_audio,
    ref_text,
    gen_text,
    remove_silence,
    seed,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1.0,
):
    if not ref_audio:
        gr.Warning("Please provide reference audio.")
        return None, None, ref_text, gen_text
    if seed is None or seed < 0 or seed > 2**31 - 1:
        seed = np.random.randint(0, 2**31 - 1)
    torch.manual_seed(int(seed))
    if not gen_text or not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return None, None, ref_text, gen_text
    if not ref_text or not ref_text.strip():
        gr.Warning("Please provide reference text.")
        return None, None, ref_text, gen_text
    processed_ref_text = process_text_with_accent(ref_text, accentizer)
    processed_gen_text = process_text_with_accent(gen_text, accentizer)
    try:
        model = ensure_model()
    except Exception as e:
        gr.Warning(f"Failed to load model: {e}")
        return None, None, processed_ref_text, processed_gen_text
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if device.type == "cuda":
            try:
                model.to(device)
                vocoder.to(device)
            except Exception as e:
                print("Warning: failed to move model/vocoder to cuda:", e)
        try:
            ref_audio_proc, processed_ref_text_final = preprocess_ref_audio_text(
                ref_audio,
                processed_ref_text,
                show_info=gr.Info
            )
        except Exception as e:
            gr.Warning(f"Preprocess failed: {e}")
            traceback.print_exc()
            return None, None, processed_ref_text, processed_gen_text
        try:
            final_wave, final_sample_rate, combined_spectrogram = infer_process(
                ref_audio_proc,
                processed_ref_text_final,
                processed_gen_text,
                model,
                vocoder,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                speed=speed,
                show_info=gr.Info,
                progress=gr.Progress(),
            )
        except Exception as e:
            gr.Warning(f"Infer failed: {e}")
            traceback.print_exc()
            return None, None, processed_ref_text, processed_gen_text
        if remove_silence:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
                    temp_path = f.name
                    sf.write(temp_path, final_wave, final_sample_rate)
                    remove_silence_for_generated_wav(temp_path)
                    final_wave_tensor, _ = torchaudio.load(temp_path)
                    final_wave = final_wave_tensor.squeeze().cpu().numpy()
            except Exception as e:
                print("Remove silence failed:", e)
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", **tempfile_kwargs) as tmp_spectrogram:
                spectrogram_path = tmp_spectrogram.name
                save_spectrogram(combined_spectrogram, spectrogram_path)
                try:
                    with Image.open(spectrogram_path) as im:
                        spectrogram_image = np.array(im.convert("RGB"))
                except Exception as e:
                    print("Load spectrogram failed:", e)
                    spectrogram_image = None
        except Exception as e:
            print("Save spectrogram failed:", e)
            spectrogram_image = None
        return (final_sample_rate, final_wave), spectrogram_image, processed_ref_text_final, processed_gen_text
    finally:
        if device.type == "cuda":
            try:
                model.to("cpu")
                vocoder.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print("Warning during cuda cleanup:", e)

with gr.Blocks(title="ESpeech-TTS") as app:
    gr.Markdown("# ESpeech-TTS")
    gr.Markdown("💡 **Совет:** Добавьте символ '+' для ударения (например, 'прив+ет')")
    gr.Markdown("❌ **Совет:** Референс должен быть не более 12 секунд")
    with gr.Row():
        with gr.Column():
            ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
            ref_text_input = gr.Textbox(
                label="Reference Text",
                lines=2,
                placeholder="Text corresponding to reference audio"
            )
        with gr.Column():
            gen_text_input = gr.Textbox(
                label="Text to Generate",
                lines=5,
                max_lines=20,
                placeholder="Enter text to synthesize..."
            )
    process_text_btn = gr.Button("✏️ Process Text (Add Accents)", variant="secondary")
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
            remove_silence = gr.Checkbox(label="Remove Silences", value=False)
        with gr.Row():
            speed_slider = gr.Slider(label="Speed", minimum=0.3, maximum=2.0, value=1.0, step=0.1)
            nfe_slider = gr.Slider(label="NFE Steps", minimum=4, maximum=64, value=48, step=2)
        cross_fade_slider = gr.Slider(label="Cross-Fade Duration (s)", minimum=0.0, maximum=1.0, value=0.15, step=0.01)
    generate_btn = gr.Button("🎤 Generate Speech", variant="primary", size="lg")
    with gr.Row():
        audio_output = gr.Audio(label="Generated Audio", type="numpy")
        spectrogram_output = gr.Image(label="Spectrogram", type="numpy")
    process_text_btn.click(
        process_texts_only,
        inputs=[ref_text_input, gen_text_input],
        outputs=[ref_text_input, gen_text_input]
    )
    generate_btn.click(
        synthesize,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            seed_input,
            cross_fade_slider,
            nfe_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input, gen_text_input]
    )

if __name__ == "__main__":
    app.launch()
