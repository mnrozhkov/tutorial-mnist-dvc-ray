import time
import random

from src.live import DVCLiveRayLogger as Live

with Live(
    dir='/tmp/dvclive',      
    dvcyaml=False, 
    save_dvc_exp=True, 
    bucket_name = "cse-cloud-version",
    s3_directory = "tutorial-mnist-dvc-ray/dvclive",
) as live: 

    # simulate training
    offset = random.uniform(0.2, 0.1)
    
    for epoch in range(5):
        fuzz = random.uniform(0.01, 0.1)
        accuracy = 1 - (2 ** - epoch) - fuzz - offset
        loss = (2 ** - epoch) + fuzz + offset

        # log metrics to studio
        live.log_metric("accuracy", accuracy)
        live.log_metric("loss", loss)

        live.next_step()
        time.sleep(3)
  