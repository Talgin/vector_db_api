## Biometrics database using FAISS, ONNX Runtime and FastAPI
The implementation of populating new face database using [Retinaface](https://docs.openvino.ai/latest/omz_models_model_retinaface_resnet50_pytorch.html), [QMagFace](https://arxiv.org/abs/2111.13475) FastAPI, FAISS and ONNX for model inference.

### Installation
> git clone https://github.com/Talgin/vector_db_api.git
- Change settings.py to point to desired folders
- Change settings.py to use either cpu or gpu (use_cpu flag)

### Running the service
> docker-compose up -d

### Issues
Sometimes you can encounter bbox errors. One solution can be to:
  - Go to rcnn/cython and do (you have to have Cython package installed):
  > python setup.py build_ext --inplace

### CHANGE HISTORY (started this in 16.12.2023)
- 16.12.2022 - function to get embeddings from a given folder

### Main idea
I will be given a list of new unique_ids, but in this list there also exist old ids. I have to keep old (non-changed) ids 
and vectorize new ones. So, I have to get new ids from the given list, by comparing it with my old ones. 

### TO-DO
- [ ] Get New IDs and subtract Old ones to get list of unique ids that you have to vectorize
- [ ] Function to get embeddings from a given folder of face images
- [ ] Function to save embeddings in chunks of 10000 in pickle files (unique_id, vector) in a /PICKLES/date folder
- [ ] Function to create FAISS index by reading chunked (10000) embeddings from some folder
- [ ] Function to create FAISS index from two given folders (previous and current) - include in a final.index only those that are in a new table
- [ ] Function to get embeddings from a given paths of face images
- [ ] Add docker images to docker hub and update readme
- [ ] Create documentation (dev, user)
- [ ] Try ScaNN
- [ ] Finish unit-tests
- [ ] Write comments for each function
- [ ] Refine code (object reusability, client creation, database connection, configs)
- [ ] Add Metadata and Docs descriptions according to [FastAPI Docs](https://fastapi.tiangolo.com/tutorial/metadata/)
- [ ] Add scaNN search functionality
- [ ] List all licenses in one file
- [ ] Connect with MLOps pipeline
