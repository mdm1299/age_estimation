import os
import time
import torch

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

            for inputs, (person, age) in loaders[phase]:
                inputs, person, age = inputs.to(device), person.to(device), age.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    output = model(inputs)
                    loss = criterion(output, (person, age))

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item()

            avgLoss = running_loss / len(loaders[phase])
            epoch_loss[phase].append(avgLoss)
            elapsed    = (time.time() - tick)
            secs, mins = int(elapsed // 60), int(elapsed % 60)
            
            print(f"{phase} Loss: {avgLoss:.4f}")
            print(f"⏱ {secs:02}:{mins:02}")
    
    if save_path:
        saveModel(model, save_path)

    return epoch_loss
            
                

                