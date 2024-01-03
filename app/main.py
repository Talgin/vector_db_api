from fastapi import FastAPI, BackgroundTasks, Response, Form, status
import settings
import uuid
import os
from datetime import datetime
from utils.vectorization import Vectorization
from pathlib import Path

app = FastAPI()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent
print(settings.BASE_DIR)

@app.post("/faiss/get_folder_embeddings", status_code=200)
async def get_folder_embeddings(response: Response, background_tasks: BackgroundTasks, new_photos_folder: str = Form(...)):
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
    if not os.path.exists(os.path.join(settings.UPDATE_PHOTOS_DIR, new_photos_folder)):
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'status': 'error', 
                'message': 'No folder {}'.format(new_photos_folder)}
    else:
        input_photos_folder = os.path.join(settings.UPDATE_PHOTOS_DIR, new_photos_folder)

        # delta_set = new_set - old_set
    
        # print('Files:', len(old_path_set), len(new_path_set)) # delete this after tests
        images_amount = len(os.listdir(input_photos_folder))

        # Create output_pickles_folder to save results of todays runs: logs, created indexes, and pickle files
        output_pickles_folder = os.path.join(settings.UPDATE_PICKLES_DIR, datetime.now().strftime("%d%m%Y"))
        if not os.path.exists(output_pickles_folder):
            os.makedirs(output_pickles_folder)

        vectorizer = Vectorization(settings.DET_MODEL, settings.REC_MODEL, settings.DET_THRESHOLD, settings.IMG_SIZE, settings.MIN_HEAD_SIZE)
        background_tasks.add_task(vectorizer.vectorize_delta_folder, input_photos_folder, output_pickles_folder, settings.VECTORS_PER_PICKLE)
        
        # send_start_time logs to redis

        return {'status': 'success', 
                'message': 'Sent to process', 
                'input_images_folder': input_photos_folder, 
                'output_folder': output_pickles_folder, 
                'images_to_process': images_amount}


@app.post("/faiss/get_folder_embeddings", status_code=200)
async def get_folder_embeddings(response: Response, background_tasks: BackgroundTasks, new_photos_folder: str = Form(...)):
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





