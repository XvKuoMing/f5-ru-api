Streaming test with OpenAI SDK
================================

Run the FastAPI app:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then run the test script:

```bash
python tests/test_openai_stream.py --host http://localhost:8000 --voice vodovoz --text "Привет, это тест потока"
```