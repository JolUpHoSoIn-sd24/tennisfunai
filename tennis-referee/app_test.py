from typing import Union

from fastapi import FastAPI, WebSocket

import io
from PIL import Image

app = FastAPI()

@app.websocket("/referee")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    index = 0
    while True:
        # data = await websocket.receive_text()
        frame_bytes = await websocket.receive_bytes()
        img = Image.open(io.BytesIO(frame_bytes))
        img.save(f'./test_results/{index}.png',"PNG")

        # await websocket.send_text(f"Message text was: {data}")