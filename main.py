from fastapi import FastAPI, BackgroundTasks, Response, Form, status
import settings
import uuid
import os
from datetime import datetime

app = FastAPI()

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
async def process_folder(response: Response, background_tasks: BackgroundTasks, old_folder: str = Form(...), new_folder: str = Form(...)):
    old_path_set = set(os.listdir(os.path.join(settings.ALL_PHOTOS_FOLDER, old_folder)))
    new_path_set = set(os.listdir(os.path.join(settings.ALL_PHOTOS_FOLDER, new_folder)))

    delta_set = new_path_set - old_path_set

    
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