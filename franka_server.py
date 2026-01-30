from franka_controller import FrankaController
import asyncio
import websockets
import json
import numpy as np

class FrankaServer:
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.controller = FrankaController(self.ip_address)
        R, t = self.controller.ee_pose()
        self.targ_pos = t

    async def handler(self, websocket):
        try:
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "set":
                    self.targ_pos = np.array(data["targ_pos"])
                elif data["type"] == "get":
                    R, t = self.controller.ee_pose()
                    response = {
                        "ee_pos": t.tolist(),
                        "targ_pos": self.targ_pos.tolist()
                    }
                    await websocket.send(json.dumps(response))

                elif data["type"] == "setq":
                    q_targ = np.array(data["q_targ"])
                    self.controller.apply_q(q_targ, duration=2.0)
                    R, t = self.controller.ee_pose()
                    self.targ_pos = t.copy()
                    response = {
                        "ee_pos": t.tolist(),
                        "targ_pos": self.targ_pos.tolist()
                    }
                    await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            print("Handler terminated")

    async def run(self):
        try:
            async with websockets.serve(self.handler, "0.0.0.0", 8765):
                print("Franka Server is running on ws://0.0.0.0:8765")
                while True:
                    try:
                        self.controller.apply_targ_pos(self.targ_pos)
                    except Exception as e:
                        print(f"Error applying target position: {e}")
                        self.controller = FrankaController(self.ip_address)
                    await asyncio.sleep(0)
        except KeyboardInterrupt:
            print("Shutting down Franka Server.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start Franka Server")
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="IP address of the Franka Emika Panda robot")
    args = parser.parse_args()

    server = FrankaServer(args.ip)
    asyncio.run(server.run())