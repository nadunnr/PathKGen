from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
import torch
from PIL import Image

model_cfg = 'conch_ViT-B-16'
#4checkpoint_path = '/checkpoints/CONCH/pytorch_model.bin'
checkpoint_path = "/home/nadun/hdd/literature-guided-digital-histopathology/checkpoints/conch/pytorch_model.bin"

def conch_encode_img(checkpoint_path, image_path, image_only, model_cfg='conch_ViT-B-16'):

    ### Load the model "create_model_from_pretrained"
    #By default, the model preprocessor uses 448 x 448 as the input size. To specify a different image size (e.g. 336 x 336), use the **force_img_size** argument.
    #You can specify a cuda device by using the **device** argument, or manually move the model to a device later using **model.to(device)**.

    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)

    # model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, force_img_size=224, device='cuda:2')
    _ = model.eval()

    ### Embed images 
    #The **.encode_image()** method encodes a batch of images into a batch of image embeddings. Note that this function applies the contrastive learning
    # projection head to the image and performs l2-normalization before returning the embedding, which is used for computing the similarity 
    # scores such as between images and texts. 
    
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        if image_only:
            #For image-only tasks, it is common to directly use the representation before the projection head and l2-normalization.
            # This is done by setting **proj_contrast=False** and **normalize=False**.
            image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
        else:
            image_embs = model.encode_image(image)

    print(f"image embeddings: {image_embs.shape}")

    return image_embs
 

def conch_encode_text(checkpoint_path, texts, model_cfg='conch_ViT-B-16'):

    #For image-only tasks, it is common to directly use the representation before the projection head and l2-normalization.
    # This is done by setting **proj_contrast=False** and **normalize=False**.
    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)

    # model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, force_img_size=224, device='cuda:2')
    _ = model.eval()

    ### Embed texts
    #The **.encode_text()** method encodes a batch of texts into a batch of l2-normalized text embeddings used for computing the similarity scores such as between images and texts. 

    texts = ["H&E image of lung adenocarcinoma. photomicrograph of a lung squamous cell carcinoma, H&E stain"]
    tokenizer = get_tokenizer() # load tokenizer
    text_tokens = tokenize(texts=texts, tokenizer=tokenizer) # tokenize the text
    text_embs = model.encode_text(text_tokens)

    print(f"text embeddings: text_embs.shape", text_embs.shape)

    return text_embs

#conch_encode_img(checkpoint_path, image_path='../data/roi1.jpg', image_only=False, model_cfg=model_cfg)
conch_encode_text(checkpoint_path, texts=None, model_cfg=model_cfg)