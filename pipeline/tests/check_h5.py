import h5py
import sys

file_path = "/home/nadun/wd/vlm_histo/data/trident_processed/exp1/20x_512px_0px_overlap/slide_features_titan/TCGA-A1-A0SB-01Z-00-DX1.B34C267B-CAAA-4AB6-AD5C-276C26F997A1.h5"

print(f"Checking file: {file_path}")
print(f"h5py version: {h5py.__version__}")

try:
    with h5py.File(file_path, 'r') as h5_file:
        print("\n✓ File opened successfully!")
        print(f"Keys in file: {list(h5_file.keys())}")
        
        for key in h5_file.keys():
            dataset = h5_file[key]
            print(f"\nDataset '{key}':")
            print(f"  Shape: {dataset.shape}")
            print(f"  DType: {dataset.dtype}")
            print(f"  Attributes: {dict(dataset.attrs)}")
            
except Exception as e:
    print(f"\n✗ Error: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Try to get more info
    try:
        import os
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        # Try to read raw bytes
        with open(file_path, 'rb') as f:
            header = f.read(8)
            print(f"File header (first 8 bytes): {header.hex()}")
            print(f"Expected HDF5 signature: 89484446 (‰HDF)")
            
    except Exception as e2:
        print(f"Could not get file info: {e2}")
    
    sys.exit(1)
