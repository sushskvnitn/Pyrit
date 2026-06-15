import pyrit

MODEL_NAME = "ollama/gemma2b"


def test_ollama_gemma2b(prompt: str) -> str:
    try:
        model = pyrit.load_model(MODEL_NAME)
        response = model.generate(prompt)
    except AttributeError:
        client = pyrit.Client()
        response = client.generate(model=MODEL_NAME, prompt=prompt)
    except Exception as exc:
        return f"Error testing {MODEL_NAME}: {exc}"

    if hasattr(response, "text"):
        return response.text
    return str(response)


if __name__ == "__main__":
    prompt = "Hello, Gemma 2b. Respond with a short confirmation message."
    output = test_ollama_gemma2b(prompt)
    print(output)