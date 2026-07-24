import uvicorn

from AI_service.main import app


if __name__ == "__main__":
    uvicorn.run("AI_service.main:app", host="127.0.0.1", port=8000, reload=False)
