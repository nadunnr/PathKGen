from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
import torch
from PIL import Image

### Load the model "create_model_from_pretrained"
#By default, the model preprocessor uses 448 x 448 as the input size. To specify a different image size (e.g. 336 x 336), use the **force_img_size** argument.
#You can specify a cuda device by using the **device** argument, or manually move the model to a device later using **model.to(device)**.

model_cfg = 'conch_ViT-B-16'
checkpoint_path = './checkpoints/CONCH/pytorch_model.bin'
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
# model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, force_img_size=224, device='cuda:2')
_ = model.eval()

### Embed images 
#The **.encode_image()** method encodes a batch of images into a batch of image embeddings. Note that this function applies the contrastive learning projection head to the image and performs l2-normalization before returning the embedding, which is used for computing the similarity scores such as between images and texts. 

image = Image.open('../docs/roi1.jpg')
image = preprocess(image).unsqueeze(0)
print(image.shape)

with torch.inference_mode():
    image_embs = model.encode_image(image)
    
print(image_embs.shape)
print(image_embs.norm(dim=-1))

#For image-only tasks, it is common to directly use the representation before the projection head and l2-normalization. This is done by setting **proj_contrast=False** and **normalize=False**.

with torch.inference_mode():
    image_embs = model.encode_image(image, proj_contrast=False, normalize=False)

print(image_embs.shape)
print(image_embs.norm(dim=-1))

### Embed texts
#The **.encode_text()** method encodes a batch of texts into a batch of l2-normalized text embeddings used for computing the similarity scores such as between images and texts. 

texts = ["H&E image of lung adenocarcinoma",
         "photomicrograph of a lung squamous cell carcinoma, H&E stain"]
tokenizer = get_tokenizer() # load tokenizer
text_tokens = tokenize(texts=texts, tokenizer=tokenizer) # tokenize the text
text_embs = model.encode_text(text_tokens)
print(text_embs.shape)