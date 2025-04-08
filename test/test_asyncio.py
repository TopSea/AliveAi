import asyncio


async def task():
    print("执行任务")


async def main():
    print("程序开始")
    future = asyncio.ensure_future(task())
    print("1")
    print("2")
    print("3")
    print("4")
    await asyncio.sleep(3)
    print("5")
    await future
    print("程序结束")


asyncio.run(main())
