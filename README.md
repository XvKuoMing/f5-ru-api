Run a local OpenAI-compatible TTS server for your F5-TTS model.

Quickstart:

1) Copy `.env.example` to `.env` and adjust paths (especially `VOICES_DIR`).

2) Prepare voices:

   voices/
     default/
       ref.wav
       ref.txt
     voice2/
       ref.wav
       ref.txt

3) Start API:

   uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
- POST /v1/audio/speech: { model, voice, input, response_format, speed, stream }
- POST /v1/audio/speech/batch: { items: [{voice, input}] }
- GET /v1/models, GET /v1/voices, GET /healthz
