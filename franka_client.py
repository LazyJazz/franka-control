import websockets
import json
import numpy as np
import asyncio
import time

class FrankaClient:
    def __init__(self, ip_address, port):
        self.uri = f"ws://{ip_address}:{port}"
        self.websocket = None

    async def ensure_connect(self):
        if self.websocket is None or self.websocket.closed:
            self.websocket = await websockets.connect(self.uri)


    async def get_pos(self):
        await self.ensure_connect()
        message = json.dumps({"type": "get"})
        await self.websocket.send(message)
        response = await self.websocket.recv()
        data = json.loads(response)
        return np.array(data["ee_pos"]), np.array(data["targ_pos"])
    
    async def set_pos(self, targ_pos):
        await self.ensure_connect()
        message = json.dumps({"type": "set", "targ_pos": targ_pos.tolist()})
        await self.websocket.send(message)

async def main():
    client = FrankaClient("localhost", 8765)
    last_tp = time.time()
    dur = 0.0
    while True:
        ee_pos, targ_pos = await client.get_pos()
        print(f"End-effector position: {ee_pos}, Target position: {targ_pos}")
        cur_tp = time.time()
        dur = cur_tp - last_tp
        last_tp = cur_tp
        new_targ_pos = targ_pos + np.array([0.0, 0.0, 0.01]) * dur
        await client.set_pos(new_targ_pos)
        time.sleep(0)

if __name__ == "__main__":
    asyncio.run(main())