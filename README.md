## Biometrics database using FAISS, ONNX Runtime and FastAPI
The implementation of populating new face database using [Retinaface](https://docs.openvino.ai/latest/omz_models_model_retinaface_resnet50_pytorch.html), [QMagFace](https://arxiv.org/abs/2111.13475) FastAPI, FAISS and ONNX for model inference.

### Installation
> git clone https://github.com/Talgin/vector_db_api.git
- Change settings.py to point to desired folders
- Change settings.py to use either cpu or gpu (use_cpu flag)

### Running the service
- Change PG_SERVER from settings.py to point to 127.0.0.1
- Check that trained_all.index file is in TRAINED_INDEX_PATH (settings.py)
- Check that there exist every path assigned in settings.py
> docker-compose up -d

### Using the service
- Place your photos into UPDATE_PHOTOS_DIR/date_of_running, e.g. UPDATE_PHOTOS_DIR/08012024
- Use /faiss/get_folder_embeddings by putting into new_photos_dir the name of of the directory where you placed photos (not whole path)
- To create new index pass the name of the photo directory to /faiss/create_new_index

### Issues
Sometimes you can encounter bbox errors. One solution can be to:
  - Go to rcnn/cython and do (you have to have Cython package installed):
  > python setup.py build_ext --inplace

### CHANGE HISTORY (started this in 16.12.2023)
- 16.12.2022 - function to get embeddings from a given folder (name of the folder better be date)
- 01.01.2024 - function to create index from given directory with pickles from different iterations
- 02.01.2024 - function to get a list of ids from table and create a new faiss index according to it using previous pickles

### Main idea
I will be given a table with mixed new and old people. I have to be given folder with new images to get embeddings. 
At the end we should have updated table and updated according to this table faiss index.

### TO-DO
- [x] Function to get embeddings from a given folder of face images
- [x] Function to save embeddings in chunks of 10000 in pickle files (unique_id, vector) in a /PICKLES/date folder
- [x] Function to create FAISS index by reading chunked (10000) embeddings from some folder
- [x] Function to create FAISS index from given directory with pickles - include in a final index only those that are in a new table
- [x] Add docker images to docker hub and update readme
- [ ] Create documentation (dev, user)
- [ ] Try ScaNN
- [ ] Finish unit-tests
- [x] Write comments to each function
- [ ] Refine code (object reusability, client creation, database connection, configs)
- [ ] Add Metadata and Docs descriptions according to [FastAPI Docs](https://fastapi.tiangolo.com/tutorial/metadata/)
- [ ] Add scaNN search functionality
- [ ] List all licenses in one file
- [ ] Connect with MLOps pipeline
