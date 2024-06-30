import torch
import config
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder):
    x,y = next(iter(val_loader))
    x = x.to(config.DEVICE)
    y = y.to(config)
    gen.eval()

    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, folder+f"/y_gen_{epoch}.png")
        save_image(x, folder+f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y, folder+f"/target.png")
    gen.train()

# def save_some_examples(gen, val_loader, epoch, folder):

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have the learning rate of the checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr