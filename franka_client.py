import websockets
import json
import numpy as np
import asyncio
import time

import pygame

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

    async def reset(self):
        await self.ensure_connect()
        message = json.dumps({"type": "setq", "q_targ": [-0.08129526674747467, -0.09338368475437164, 0.02063392661511898, -2.354853630065918, 0.002519397297874093, 2.2613837718963623, 0.723608493804932]})
        await self.websocket.send(message)
        response = await self.websocket.recv()
        data = json.loads(response)
        return np.array(data["ee_pos"]), np.array(data["targ_pos"])
    
    async def close(self):
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

async def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="IP address")
    parser.add_argument("--port", type=int, default=8765, help="Port number")
    args = parser.parse_args()

    client = FrankaClient(args.ip, args.port)
    last_tp = time.time()
    dur = 0.0

    pygame.init()
    pygame.joystick.init()

    while True:
        ee_pos, targ_pos = await client.get_pos()
        print(f"End-effector position: {ee_pos}, Target position: {targ_pos}")
        cur_tp = time.time()
        dur = cur_tp - last_tp
        last_tp = cur_tp

        move_vel = np.array([0.0, 0.0, 0.0])

        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            
            pygame.event.pump()  # Process event queue
            
            axis_0 = joystick.get_axis(0)
            axis_1 = joystick.get_axis(1)

            axis_2 = (joystick.get_axis(2) + 1.0) * 0.5
            axis_5 = (joystick.get_axis(5) + 1.0) * 0.5
            move_vel[1] = axis_0 * 0.05  # Scale to max 0.1 m/s
            move_vel[0] = axis_1 * 0.05  # Invert Y axis
            move_vel[2] = (axis_5 - axis_2) * 0.05  # Scale to max 0.1 m/s

              # if button 1 pressed, terminate

            if joystick.get_button(3):
                ee_pos, targ_pos = await client.reset()
                print(f"Robot reset. End-effector position: {ee_pos}, Target position: {targ_pos}")

            if joystick.get_button(1):
                print("Exiting joystick control.")
                await client.close()
                break

        new_targ_pos = targ_pos + move_vel * dur
        print(f"Setting new target position: {new_targ_pos}")
        await client.set_pos(new_targ_pos)

if __name__ == "__main__":
    asyncio.run(main())