from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

# Trained index path
TRAINED_INDEX_PATH = '/TENGRI/STORAGE/trained_index/trained_all.index'

# Directory to place newly obtained photos DIR/new_photos_dir
UPDATE_PHOTOS_DIR = '/TENGRI/STORAGE/update_photos'

# Directory to store chunks of pickle files with newly obtained embeddings
UPDATE_PICKLES_DIR = '/TENGRI/STORAGE/update_pickles'

# Directory to store newly created FAISS index
UPDATE_FAISS_DIR = '/TENGRI/STORAGE/update_indexes'

# Amount of id:vector pairs to be stored in one pickle file
VECTORS_PER_PICKLE = 10000

# Settings of detector and feature extractor (recognition)
DET_MODEL = '/api_folder/models/detection/R50'
REC_MODEL ='/api_folder/models/recognition/model,0'

DET_THRESHOLD = 0.95
IMG_SIZE = '112,112'
MIN_HEAD_SIZE = 30