from fastapi import FastAPI, BackgroundTasks, Response, Form, status
import settings
import uuid
import os
from datetime import datetime
from utils.vectorization import Vectorization
from fs.faisser import Faisser
from pathlib import Path
import pickle

app = FastAPI()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent
print(settings.BASE_DIR)
vectorizer = Vectorization(settings.DET_MODEL, settings.REC_MODEL, settings.DET_THRESHOLD, settings.IMG_SIZE, settings.MIN_HEAD_SIZE)
fs_worker = Faisser(settings.UPDATE_PICKLES_DIR, settings.UPDATE_FAISS_DIR)

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
        output_pickles_dir = os.path.join(settings.UPDATE_PICKLES_DIR, datetime.now().strftime("%d%m%Y"))
        if not os.path.exists(output_pickles_dir):
            os.makedirs(output_pickles_dir)

        background_tasks.add_task(vectorizer.vectorize_delta_dir, input_photos_dir, output_pickles_dir, settings.VECTORS_PER_PICKLE)
        
        # send_start_time logs to redis

        return {'status': 'success', 
                'message': 'Sent to process', 
                'input_images_folder': input_photos_dir, 
                'output_folder': output_pickles_dir, 
                'images_to_process': images_amount}


@app.post("/faiss_create_new_index", status_code=200)
async def faiss_create_new_index(response: Response, background_tasks: BackgroundTasks, dir_with_pickles: str = Form(...)):
    """Obtaining the embeddings from the given directory with delta images

    Parameters
    ----------
    dir_with_pickles : str
        Name of the directory with pickles obtained in get_folder_embeddings

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
    # Read ids from unique_ud_gr table to save only listed ids in a final index


    
    output_pickles_dir = os.path.join(settings.UPDATE_PICKLES_DIR, dir_with_pickles)
    if os.path.exists(output_pickles_dir):
        # Read previously obtained pickles
        fs_worker.read_pickles(settings.UPDATE_PICKLES_DIR)

        try:
            result = db_worker.create_block_and_index(new_vectors, new_ids_np, configs.TRAINED_INDEX_PATH, 
                                                    configs.PATH_TO_SAVE_NEW_BLOCK, configs.MERGED_INDEX_PATH, 
                                                    configs.LOCAL_FAISS_BACKUP_PATH)
        except:
            result = False
        if result['status'] == True:
            return {'status': 'success', 'index-size': result['size']}
        else:
            return {'status': 'error', 'message': 'Could not save new faiss ' + configs.LOCAL_FAISS_BACKUP_PATH}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'status': 'error', 'message': 'Could not save new faiss ' + configs.LOCAL_FAISS_BACKUP_PATH}





