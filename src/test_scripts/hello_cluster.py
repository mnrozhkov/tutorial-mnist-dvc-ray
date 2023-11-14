import ray

@ray.remote
def hello_world():
    return "Hello Ray cluster"

# Automatically connect to the running Ray cluster.
ray.init()
print(ray.get(hello_world.remote()))