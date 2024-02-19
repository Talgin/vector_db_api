from fastapi import FastAPI, BackgroundTasks, Response, Form, status
import settings
import os
from datetime import datetime
from utils.vectorization import Vectorization
from fs.faisser import Faisser
from pathlib import Path
import numpy as np

app = FastAPI()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

vectorizer = Vectorization(settings.DET_MODEL, settings.REC_MODEL, settings.DET_THRESHOLD, settings.IMG_SIZE, settings.MIN_HEAD_SIZE)
fs_worker = Faisser(settings.UPDATED_PICKLES_DIR)

@app.post("/faiss/get_folder_embeddings", status_code=200)
async def get_folder_embeddings(response: Response, background_tasks: BackgroundTasks, new_photos_dir: str = Form(...)):
    """Obtaining the embeddings from the given directory with delta images

    Parameters
    ----------
    new_photos_folder : str
        Name of the directory with images

    Returns
    -------
    status
        success or error
    message
        message explaining the status
    input_images_folder
        name of the input images folder
    output_folder
        name of the output folder
    images_to_process
        amount of images to be processed
    """
    if not os.path.exists(os.path.join(settings.UPDATE_PHOTOS_DIR, new_photos_dir)):
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'status': 'error', 
                'message': 'No folder {}'.format(new_photos_dir)}
    else:
        input_photos_dir = os.path.join(settings.UPDATE_PHOTOS_DIR, new_photos_dir)

        # delta_set = new_set - old_set
    
        # print('Files:', len(old_path_set), len(new_path_set)) # delete this after tests
        images_amount = len(os.listdir(input_photos_dir))

        # Create output_pickles_folder to save results of todays runs: logs, created indexes, and pickle files
        output_pickles_dir = os.path.join(settings.UPDATED_PICKLES_DIR, datetime.now().strftime("%d%m%Y"))
        if not os.path.exists(output_pickles_dir):
            os.makedirs(output_pickles_dir)

        background_tasks.add_task(vectorizer.vectorize_delta_dir, input_photos_dir, output_pickles_dir, settings.VECTORS_PER_PICKLE)
        
        # send_start_time logs to redis

        return {'status': 'success', 
                'message': 'Sent to process', 
                'input_images_folder': input_photos_dir, 
                'output_folder': output_pickles_dir, 
                'images_to_process': images_amount}


@app.post("/faiss/create_new_index", status_code=200)
async def faiss_create_new_index(response: Response, background_tasks: BackgroundTasks, new_dir_name: str = Form(...)):
    """Create new index from new ids and vectors using records list from PostgreSQL database

    Parameters
    ----------
    new_dir_name : str
        Name of the directory with pickles obtained in get_folder_embeddings (output_pickles_dir)

    Returns
    -------
    status
        success or error
    message
        message explaining the status
    new_dir_name
        name of the directory where all info of the current deltas stored
    index-size
        size of the newly created faiss index
    """
    # Read ids from unique_ud_gr table to save only listed ids in a final index
    records_from_db = fs_worker.read_ids_from_postgres_db(settings.PG_SERVER, settings.PG_PORT, settings.PG_DB, 
                                                          settings.PG_USER, settings.PG_PASS, settings.PG_SCHEMA_AND_TABLE)
    records_from_db = [elem[0] for elem in records_from_db]

    updated_pickles_dir = os.path.join(settings.UPDATED_PICKLES_DIR, new_dir_name)
    if os.path.exists(updated_pickles_dir):
        # Run in the background because it will cause socket hang up Error
        background_tasks.add_task(create_index, new_dir_name, records_from_db)
        return {'status': 'success', 
                'message': 'Sent to process', 
                'Amount of records (DB)': len(records_from_db), 
                'Pickles dir': updated_pickles_dir}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'status': 'error', 
                'message': 'Could not find pickles in ' + updated_pickles_dir}


def create_index(new_dir_name, records_from_db):
    # Read previously obtained pickles
    pickles_dict = fs_worker.read_pickles()
    print('Pickles dict:', len(pickles_dict))
    # Filter ids and vectors according to the list from Postgre table
    filtered_dict = {key: pickles_dict[key] for key in records_from_db}
    # new_ids, new_vectors
    identificators = []
    vectors = []
    for k in filtered_dict.keys():
        identificators.append(k)
        vectors.append(filtered_dict[k])
    # Formatting vectors and ids
    vectors = np.array(vectors, dtype=np.float32)
    identificators = np.array(list(map(int, identificators)))
    # Directory to save new faiss index
    new_faiss_index_dir = os.path.join(settings.UPDATED_FINAL_INDEX, new_dir_name)
    if not os.path.exists(new_faiss_index_dir):
        os.makedirs(new_faiss_index_dir)
    try:
        result = fs_worker.create_block_and_index(identificators, vectors, settings.TRAINED_INDEX_PATH, new_faiss_index_dir)
    except:
        result = {'status': False}
    if result['status'] == True:
        return {'status': 'success', 'message': 'Successfully updated index', 'new_dir_name': new_dir_name, 'index-size': result['size'], 'new-index-path': result['path']}
    else:
        return {'status': 'error', 'message': 'Could not save new faiss in ' + new_faiss_index_dir}


