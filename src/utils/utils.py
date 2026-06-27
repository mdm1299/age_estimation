import os
import time
import torch
from tqdm import tqdm

def elapsedTime(lastTick):
    elapsed    = (time.time() - lastTick)
    return int(elapsed // 60), int(elapsed % 60)

def saveModel(model, save_path):
    # create the directory if it doesn't extist
    dir = os.path.dirname(save_path)
    os.makedirs(dir, exist_ok=True)
    # save the model
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

def train(model, loaders, criterion, optimizer, epochs=25, device=torch.device('cpu'), save_path=None):
    epoch_loss = {
        "train": [],
        "val": []
    }

    is_cuda = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if is_cuda else None

    model.to(device)

    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}/{epochs}")
        print("-" * 15)

        for phase in ["train", "val"]:
            running_loss = 0
            tick = time.time()

            if phase == "train":
                model.train()
            else:
                model.eval()

            for batchIdx, (inputs, (person, age)) in (
                pbar := tqdm(enumerate(loaders[phase]), total=len(loaders[phase]))
            ):
                inputs, person, age = inputs.to(device), person.to(device), age.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if is_cuda:
                        with torch.autocast("cuda"):
                            output = model(inputs)
                            loss   = criterion(output, (person, age))
                    else:
                        output = model(inputs)
                        loss   = criterion(output, (person, age))

                    if phase == "train":
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item()

                pbar.set_description(
                    f"[{epoch:02} | {epochs:02}] Loss: {loss.item():.4f}"
                )

                if (batchIdx+1) % 500 == 0:             # checkpoint every 500 batches
                    saveModel(model, save_path)
                    saveModel(optimizer, save_path.removesuffix(".pt") + "_optim.pt")

            avgLoss = running_loss / len(loaders[phase])
            epoch_loss[phase].append(avgLoss)
            mins, secs = elapsedTime(tick)

            print(f"{phase} Loss: {avgLoss:.4f}")
            print(f"⏱ {secs:02}:{mins:02}")

    if save_path:
        saveModel(model, save_path)
        saveModel(optimizer, save_path.removesuffix(".pt") + "_optim.pt")

    return epoch_loss


def extract_features(model, loader, shard_dir, split, device, shard_size=256):
    def flush_shard(buf, id):
        path = os.path.join(shard_dir, f"{split}_shard_{id:04d}.pt")
        if not os.path.exists(path):
            torch.save({"features": torch.stack(buf)}, path)

    os.makedirs(shard_dir, exist_ok=True)
 
    all_shard_ids = []
    all_local_ids = []
    all_persons   = []
    all_ages      = []
    shard_id     = 0
    shard_buffer = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for clips, (persons, ages) in tqdm(loader, desc=f"Extracting [{split}]"):
            clips    = clips.to(device)
            features = model(clips).cpu()
 
            for i in range(len(features)):
                all_shard_ids.append(shard_id)
                all_local_ids.append(len(shard_buffer))
                all_persons.append(persons[i].item())
                all_ages.append(ages[i].item())
 
                shard_buffer.append(features[i])
 
                if len(shard_buffer) == shard_size:
                    flush_shard(shard_buffer, shard_id)
                    shard_buffer = []
                    shard_id    += 1
 
    if shard_buffer:
        flush_shard(shard_buffer, shard_id)
 
    torch.save({
        "shard_ids": torch.tensor(all_shard_ids, dtype=torch.long),
        "local_ids": torch.tensor(all_local_ids, dtype=torch.long),
        "persons":   torch.tensor(all_persons,   dtype=torch.float32),
        "ages":      torch.tensor(all_ages,       dtype=torch.float32),
    }, os.path.join(shard_dir, f"{split}_index.pt"))
 
    print(f"[{split}] {len(all_shard_ids)} samples, {shard_id + 1} shards in {shard_dir}/")
 