import torch, time

@torch.no_grad()
def throughput(batch_size, input_size, model, channel_last = False):

    # batch_size = args.batch_size

    images = torch.randn(batch_size, 3, input_size, input_size)
    images = images.cuda(non_blocking=True)

    if channel_last:
        images = images.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last) 

    with torch.no_grad():
        model.eval()

        # eval throughput
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
    tic2 = time.time()
    eval_throughput = 30 * batch_size / (tic2 - tic1)
    # print(f"Eval: batch_size {batch_size} throughput {eval_throughput}")
    return eval_throughput

