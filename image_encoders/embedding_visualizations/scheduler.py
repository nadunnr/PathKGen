import subprocess

slide_encoder = "feather"
patch_size = 512
subprocess.run(f"python slide_embeddings.py --encoder_name {slide_encoder} --patch_size {patch_size} ", shell=True, check=True)

slide_encoder = "titan"
patch_size = 512
subprocess.run(f"python slide_embeddings.py --encoder_name {slide_encoder} --patch_size {patch_size} ", shell=True, check=True)

slide_encoder = "madeleine"
patch_size = 256
subprocess.run(f"python slide_embeddings.py --encoder_name {slide_encoder} --patch_size {patch_size} ", shell=True, check=True)

slide_encoder = "prism"
patch_size = 224
subprocess.run(f"python slide_embeddings.py --encoder_name {slide_encoder} --patch_size {patch_size} ", shell=True, check=True)

