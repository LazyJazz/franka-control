import asyncio

async def worker(id):
    while True:
        print(f"Worker {id} is working...")
        await asyncio.sleep(0)

async def main():
    tasks = [asyncio.create_task(worker(i)) for i in range(3)]
    # run all tasks simultaneously
    while True:
        print("Main is running...")
        await asyncio.sleep(0)
        pass

if __name__ == "__main__":
    asyncio.run(main())