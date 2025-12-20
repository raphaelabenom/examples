import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

openai_api_key = os.getenv("OPENAI_API_KEY")


def main() -> None:
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    response = httpx.get(
        "https://api.openai.com/v1/models",
        headers=headers,
    )

    with open("openai_models.txt", "w", encoding="utf-8") as file:
        for k, v in enumerate(response.json()["data"]):
            file.write(f"{k}. {v['id']}\n")


if __name__ == "__main__":
    main()
