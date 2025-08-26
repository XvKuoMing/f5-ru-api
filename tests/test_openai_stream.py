import argparse
import os
import sys
import time
import wave
from typing import Iterator

try:
    from openai import OpenAI
except Exception:
    print("Please install openai: pip install openai>=1.0.0")
    sys.exit(1)

# uv run tests/test_openai_stream.py --voice vodovoz --text "Привет, это тест потока" --wav out.wav

def stream_speech(host: str, model: str, voice: str, text: str) -> Iterator[bytes]:
    client = OpenAI(
        base_url=f"{host}/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "dummy")
    )
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format="pcm16"
    ) as response:
        for chunk in response.iter_bytes():
            yield chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8080", help="Server host, e.g. http://localhost:8080")
    parser.add_argument("--model", default="ESpeech/ESpeech-TTS-1_RL-V2", help="Model name")
    parser.add_argument("--voice", required=True, help="Voice name")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--outfile", default="out.pcm", help="Output PCM file path")
    parser.add_argument("--wav", default=None, help="Optional WAV output path (24000 Hz, mono)")
    args = parser.parse_args()

    start = time.time()
    with open(args.outfile, "wb") as f:
        wav = None
        if args.wav:
            wav = wave.open(args.wav, "wb")
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit PCM
            wav.setframerate(24000)
        try:
            for i, chunk in enumerate(stream_speech(args.host, args.model, args.voice, args.text)):
                if i == 0:
                    print("Receiving stream...")
                f.write(chunk)
                if wav is not None:
                    wav.writeframes(chunk)
        finally:
            if wav is not None:
                wav.close()
    dur = time.time() - start
    extra = f" and WAV to {args.wav}" if args.wav else ""
    print(f"Done. Wrote raw PCM to {args.outfile}{extra} in {dur:.2f}s. To play: ffplay -f s16le -ar 24000 -ac 1 {args.outfile}")


if __name__ == "__main__":
    main()


