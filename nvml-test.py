import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(mem.used / 1024 / 1024)

pynvml.nvmlShutdown()