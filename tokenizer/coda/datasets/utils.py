from PIL import Image

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def _tensor2img(img_tensor, path):
    img_tensor = img_tensor * 0.5 + 0.5
    img_tensor = img_tensor.clamp(0, 1)
    img_tensor = img_tensor * 255
    img_tensor = img_tensor.detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')
    Image.fromarray(img_tensor).save(path)
