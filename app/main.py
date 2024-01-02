from fastapi import FastAPI, BackgroundTasks, Response, Form, status
import settings
import uuid
import os
from datetime import datetime
from utils.vectorization import Vectorization
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

print(settings.BASE_DIR)
'''

'''
@app.post("/faiss/process_folder/", status_code=200)
async def process_folder(response: Response, background_tasks: BackgroundTasks, folder_name: str = Form(...)):
    file_path = os.path.join(settings.ALL_PHOTOS_FOLDER, folder_name)
    print('File:', file_path) # delete this after tests
    if not os.path.exists(file_path):
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'status': 'error', 'message': 'No such folder.'}
    else:
        # Create todays_folder to save results of todays runs: logs, created indexes, and pickle files
        todays_folder = os.path.join(settings.CROPS_FOLDER, datetime.now().strftime("%d%m%Y"))
        if not os.path.exists(todays_folder):
            os.makedirs(todays_folder)

        img_name = img_name = str(uuid.uuid4())
        new_img_folder = os.path.join(todays_folder, img_name)
        if not os.path.exists(new_img_folder):
            os.makedirs(new_img_folder)

        background_tasks.add_task(process_image_folder, file_path, new_img_folder, img_name)

        return {'status': 'success', 'message': 'Sent to process', 'date_folder': todays_folder, 'unique_id': img_name, 'video_folder': new_img_folder}


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




