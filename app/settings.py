from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

# Connection option to Postgres database
PG_SERVER = '127.0.0.1' # '167.235.199.85' # change to 127.0.0.1
PG_PORT = '20005'
PG_DB = 'fr_kpp'
PG_USER = 'face_reco_admin'
PG_PASS = 'qwerty123'
PG_SCHEMA_AND_TABLE = 'public.unique_gr_code' # in test change to fr.unique_ud_test

# Trained index path
TRAINED_INDEX_PATH = '/TENGRI/STORAGE/trained_index/trained_all.index'

# Directory to place newly obtained photos DIR/new_photos_dir
UPDATE_PHOTOS_DIR = '/new_tom/UNTAR_FILES'

# Directory to store chunks of pickle files with newly obtained embeddings
UPDATED_PICKLES_DIR = '/new_tom/updated_pickles'

# Directory to store updated final indexes
UPDATED_FINAL_INDEX = '/new_tom/updated_final_index'

# Amount of id:vector pairs to be stored in one pickle file
VECTORS_PER_PICKLE = 1000 # 10000

# Settings of detector and feature extractor (recognition)
DET_MODEL = '/api_folder/models/detection/R50'
REC_MODEL ='/api_folder/models/recognition/model,0'

DET_THRESHOLD = 0.95
IMG_SIZE = '112,112'
MIN_HEAD_SIZE = 30
