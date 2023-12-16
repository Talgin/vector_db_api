from fastapi import FastAPI, BackgroundTasks, Response, Form, status
import settings
import uuid
import os
from datetime import datetime

app = FastAPI()

@app.post("/video/process_uploaded_video/", status_code=200)
async def process_uploaded_video(response: Response, background_tasks: BackgroundTasks, video_name: str = Form(...)):
    file_path = os.path.join(settings.VIDEO_UPLOAD_FOLDER, video_name)
    print('File:', file_path)
    if not os.path.exists(file_path):
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'status': 'ERROR', 'message': 'No such file.'}
    else:
        todays_folder = os.path.join(settings.CROPS_FOLDER, datetime.now().strftime("%Y%m%d"))
        if not os.path.exists(todays_folder):
            os.makedirs(todays_folder)

        img_name = img_name = str(uuid.uuid4())
        new_img_folder = os.path.join(todays_folder, img_name)
        if not os.path.exists(new_img_folder):
            os.makedirs(new_img_folder)

        background_tasks.add_task(process_video_file, file_path, new_img_folder, img_name)

        return {'status': 'OK', 'message': 'Sent to process', 'date_folder': todays_folder, 'unique_id': img_name, 'video_folder': new_img_folder}
