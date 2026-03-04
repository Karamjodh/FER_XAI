import GPUtil
import time

while True:
    g = GPUtil.getGPUs()[0]
    print(f'Temp: {g.temperature}C | Load: {g.load*100:.0f}% | VRAM: {g.memoryUsed}/{g.memoryTotal}MB')
    time.sleep(10)