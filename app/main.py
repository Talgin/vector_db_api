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


@app.post("/faiss/process_folder/", status_code=200)
async def process_folder(response: Response, background_tasks: BackgroundTasks, new_photos_folder: str = Form(...)):
    if not os.path.exists(os.path.join(settings.UPDATE_PHOTOS_FOLDER, new_photos_folder)):
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'status': 'error', 'message': 'No folder {}'.format(new_photos_folder)}
    else:
        new_photo_list = os.listdir(os.path.join(settings.UPDATE_PHOTOS_FOLDER, new_photos_folder))

        # delta_set = new_set - old_set
    
        # print('Files:', len(old_path_set), len(new_path_set)) # delete this after tests
        print('Delta:', len(new_photo_list))

        # Create todays_folder to save results of todays runs: logs, created indexes, and pickle files
        todays_folder = os.path.join(settings.ALL_PICKLES_FOLDER, datetime.now().strftime("%d%m%Y"))
        if not os.path.exists(todays_folder):
            os.makedirs(todays_folder)

        folder_name = str(uuid.uuid4())
        # new_img_folder = os.path.join(todays_folder, img_name)
        # if not os.path.exists(new_img_folder):
        #     os.makedirs(new_img_folder)

        vectorize = Vectorization(settings.DET_MODEL, settings.REC_MODEL, settings.DET_THRESHOLD, settings.IMG_SIZE, settings.MIN_HEAD_SIZE)
        vectorizer = Vectorization()
        background_tasks.add_task(process_image_folder, file_path, new_img_folder, img_name)
        # send_start_time logs to redis

        return {'status': 'success', 'message': 'Sent to process', 'Folder': todays_folder, 'unique_id': img_name, 'video_folder': new_img_folder}


