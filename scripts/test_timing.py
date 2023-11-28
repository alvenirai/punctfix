from time import time
import torch
from punctfix import PunctFixer

MODEL_INPUT = "det der sker over de tre dage fra præsident huden tav ankommer til københavn det er at der " \
                "sådan en bliver spillet sådan et form for tom og jerry kispus mellem københavns politi og " \
                "så de har danske demonstranter for tibet og fåfalungongsom meget gerne vil vise deres " \
                "utilfredshed med det kinesiske regime og det de opfatter som undertrykkelse af de her " \
                "mindretal i kine og lige nu står støttekomiteen for ti bedet bag en demonstration på" \
                " højbro plads i københavn lisbeth davidsen hvor mange er der kommet det er ikke " \
                "de store folkemasser der er mødt op her på" * 10

def time_fp(device_str: str, batch_size: int):
    print(">>> Profiling device %s on batch size %i" % (device_str, batch_size))
    start = time()
    model = PunctFixer(language="da", device=device_str, batch_size=batch_size)
    print("Initialization time %f" % (time() - start))

    # Warmup potential CUDA device
    model.punctuate(MODEL_INPUT)

    times = []
    for _ in range(5):
        start = time()
        model.punctuate(MODEL_INPUT)
        times.append(time() - start)
    print("Average time: %f\nStd. time: %f" % (torch.tensor(times).mean().item(), torch.tensor(times).std().item()))


if __name__ == "__main__":
    devices = ["cpu"]
    batch_sizes = [1, 16, 32, 64]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        for batch_size in batch_sizes:
            time_fp(device, batch_size)
