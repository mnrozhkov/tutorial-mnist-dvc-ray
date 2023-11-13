# script.py
import ray

@ray.remote
def hello_world():
    return "hello world"

# Automatically connect to the running Ray cluster.
ray.init("ray://172.31.38.195:8265")
print(ray.get(hello_world.remote()))