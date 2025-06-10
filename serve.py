from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

# Mount the templates directory
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 