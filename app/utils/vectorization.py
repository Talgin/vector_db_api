#!/usr/bin/env python
# coding: utf-8
import cv2
from facer.retinaface import RetinaFace
from facer.face_model_tengri import FaceModel
from facer.align_faces import align_img
import numpy as np
import os, sys
import time
import json
import pickle
from logbook import Logger, FileHandler, Processor, StreamHandler
import logbook
import GPUtil
import tarfile
# From settings file import settings

class Vectorization:
    """
    A class used to do face detection and obtaining embeddings from some given collection of images

    ...

    Attributes
    ----------
    detection_model : str
        location of the detection model
    recognition_model : str
        location of the recognition model
    detection_threshold : str
        threshold for face detection
    img_size : str
        size of the image that is feed to recognition model
    min_head_size : int
        minimum head size in the frame (in pixels)
    data_root : str
        root folder from where to get data (images to be processed)

    Methods
    -------
    get_logger(data_root_folder, image_folder_name, debug=True/False)
        Creates a logger object with specified parameters
    frame_preprocess(img, scales)
        Preprocessing of obtained image (resizing, cropping, aligning, etc.)
    get_vector(test_image)
        Obtaining the embeddings from the given image
    vectorize_delta_folder(image_folder)
        Gets embeddings from the collection of images
    """
    def __init__(self, detection_model, recognition_model, detection_threshold, img_size, min_head_size):
        deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.6, maxMemory=0.75, attempts=1, interval=20, verbose=False)
        self.gpu_id = None
        if len(deviceID) > 0:
            self.gpu_id = deviceID[0]
        else:
            print('No GPU available or not enough space.')
            return 0
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.scales = [900,900] # [1080,1920]
        self.detection_threshold = detection_threshold
        self.img_size = img_size
        self.ga_model = ''
        self.threshold = 1.24
        self.det_type = 0
        self.flip = 0
        self.min_head_size = min_head_size
        os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"]="0"
        os.environ["MXNET_CPU_WORKER_NTHREADS"]="24"
        os.environ["MXNET_ENGINE_TYPE"]="ThreadedEnginePerDevice"
        os.environ["MXNET_CUDNN_LIB_CHECKING"]="0"
        self.detector = RetinaFace(self.detection_model, 0, self.gpu_id, 'net3')
        self.recognitor = FaceModel(self.gpu_id, self.recognition_model, self.ga_model, self.threshold, self.det_type, self.flip, self.img_size)


    def get_logger(self, output_folder, image_folder_name):
        """Creates a logger object with given parameters and creates a folder where the current log is saved

        Parameters
        ----------
        output_folder : str
            Location of the directory to save created data
        image_folder_name : str
            Name of the currently obtained image folder (e.g. '21_07_05_16_37', 'some_dataset', etc.)

        Returns
        -------
        Logger object
            Logger object that can be used to log events
        """
        logs_folder = os.path.join(output_folder, 'logs')
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        logs_file = os.path.join(logs_folder, 'project_logs.log')
        logbook.set_datetime_format('local')
        logger = Logger(image_folder_name)
        logger.handlers.append(FileHandler(logs_file, level='DEBUG', bubble=True, mode='a', encoding='utf-8'))
        logger.handlers.append(StreamHandler(sys.stdout, level='INFO', bubble=True))
        # handler = StreamHandler(sys.stdout) if self.debug else FileHandler(logs_folder + image_folder_name + '.log', level='WARNING', mode='a', encoding='utf-8')
        # handler.push_application()
        return logger # Logger(os.path.basename(image_folder_name))


    def frame_preprocess(self, img, scales):
        """Preprocessing of obtained image (resizing, cropping, aligning, etc.)

        Parameters
        ----------
        img : np.ndarray
            Image read by opencv
        scales : str
            Default scales of the frame that is used

        Returns
        -------
        Aligned image
            head crop aligned horizontally having dimensions 112*112 pixels
        """
        thresh = self.detection_threshold
        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        faces, landmarks = self.detector.detect(img, thresh, scales=scales, do_flip=self.flip)

        aligned = None
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            # Getting the size of head rectangle
            height_y = box[3] - box[1]
            width_x = box[2] - box[0]
            # Calculating cropping area
            if landmarks is not None and height_y > self.min_head_size:
                center_y = box[1] + ((box[3] - box[1])/2)
                center_x = box[0] + ((box[2] - box[0])/2)
                rect_y = int(center_y - height_y/2)
                rect_x = int(center_x - width_x/2)
                landmark5 = landmarks[i].astype(np.int)
                aligned = align_img(img, landmark5)
        return aligned


    def get_vector(self, test_image):
        """Obtaining the embeddings from the given image

        Parameters
        ----------
        test_image : str
            Location of the image

        Returns
        -------
        feature
            embedding (array) of size (1,512)
        """
        try:
            if os.stat(test_image).st_size > 0:
                img = cv2.imread(test_image)
                crop = self.frame_preprocess(img, self.scales)
                nimg = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                aligned = np.transpose(nimg, (2,0,1))
                feature = self.recognitor.get_feature(aligned)
                return feature
            else:
                return None
        except:
            print('Bad Image')
            return None


    def vectorize_delta_dir(self, input_image_folder, output_folder, vectors_per_pickle):
        """Gets embeddings from the collection of images

        Parameters
        ----------
        input_image_folder : str
            Directory where new images were placed
        pickles_output_folder : str
            Repository to store pickles with vectors
        vectors_per_pickle : int
            Amount of vectors to store in one pickle file in a chunk of pickles

        Returns
        -------
        vector_dict
            dictionary that stores the obtained embeddings with ud_code with structure like 'ud_code': vector
        bad_cropped
            dictionary that stores names of non-processable images
        """
        log = self.get_logger(output_folder, input_image_folder)
        # IM_FOLDER = self.data_root + image_folder
        if len(os.listdir(input_image_folder)) != 0:
            PICKLES_FOLDER = os.path.join(output_folder, 'pickles')
            if not os.path.exists(PICKLES_FOLDER):
                os.makedirs(PICKLES_FOLDER)
                log_message = {'status': 'success', 'message': 'Pickles, logs folder created'}
                log.info(log_message)
            BAD_IMAGES = os.path.join(output_folder, 'jsons')
            if not os.path.exists(BAD_IMAGES):
               os.makedirs(BAD_IMAGES)
               log_message = {'status': 'success', 'message': 'BAD_IMAGES folder created'}
               log.info(log_message)
            log_message = {'status': 'success', 'message': 'Images amount in IM_FOLDER obtained.', 'amount': len(os.listdir(input_image_folder))}
            log.info(log_message)

            vector_dict = {}
            image_counter = 0
            bad_cropped = {"ud_code":{}}
            bad_counter = 0

            start = time.time()

            # Main loop
            for image in os.listdir(input_image_folder):
                #print(image)
                # Getting current image from folder
                current = os.path.join(input_image_folder, image)
                # print(current)
                # Obtaining ud_code value to store it as an ID
                ud_code = image.split('.')[0]
                try:
                    # Obtaining embedding from and image
                    vector = self.get_vector(current)
                    if vector is not None and len(vector) > 0:
                        vector_dict[ud_code] = list(vector)
                        image_counter += 1
                        if image_counter!= 0 and image_counter%vectors_per_pickle==0:
                            if bool(vector_dict):
                                with open(PICKLES_FOLDER + '/' + str(image_counter) + '.pickle', 'wb') as handle:
                                    pickle.dump(vector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                # print('All pickles are written', len(vector_dict))
                                log_message = {'status': 'success', 'message': 'PICKLES_ARE_WRITTEN', 'AMOUNT': len(vector_dict)}
                                log.info(log_message)
                                vector_dict = {}
                            log_message = {'status': 'success', 'processed_images': image_counter}
                            log.info(log_message)
                    else:
                        bad_cropped["ud_code"][bad_counter] = ud_code
                        bad_counter += 1
                        log_message = {'status': 'warning', 'message':'BAD_IMAGE', 'image_no': ud_code}
                        log.info(log_message)
                except:
                    bad_cropped[bad_counter] = ud_code
                    bad_counter += 1
                    log_message = {'status': 'warning', 'message':'BAD_IMAGE', 'image_no': ud_code}
                    log.info(log_message)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # If there exist embeddings in vector_dict we have to write them into our pickle file
            if bool(vector_dict):
                with open(PICKLES_FOLDER + '/' + str(image_counter) + '.pickle', 'wb') as handle:
                    pickle.dump(vector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # print('All pickles are written', len(vector_dict))
                log_message = {'status': 'success', 'message': 'LAST_PICKLES_ARE_WRITTEN', 'AMOUNT': len(vector_dict)}
                log.info(log_message)
            if bool(bad_cropped):
                with open(BAD_IMAGES + '/' + 'errors.json', 'a') as f:
                    f.write(json.dumps(bad_cropped))
                log_message = {'status': 'success', 'message': 'BAD_IMAGES_LIST_WRITTEN', 'AMOUNT': len(bad_cropped)}
                log.info(log_message)

            end = time.time()
            log_message = {'status': 'finished', 'message': 'DELTA_PROCESSING_FINISHED', 'processed_amount': len(os.listdir(input_image_folder)), 'time_spent': end - start}
            log.info(log_message)

            return vector_dict
        else:
            log_message = {'status': 'error', 'message': 'NO_IMAGE_IN_SOURCE_FOLDER', 'processed_amount': len(os.listdir(input_image_folder))}
            log.info(log_message)
            return None


    def create_tar(self, image_folder):
        folder_list = None
        # if self.debug:
        #     folder_list = ['pickles', 'test_photo', 'jsons']
        # else:
        #     folder_list = ['pickles', 'photo', 'jsons']
        tar = tarfile.open(self.data_root + image_folder + '/' + image_folder + '.tar.gz', 'w:gz')
        try:
            for file in folder_list:
                if file == "photo":
                    current_file = self.data_root + image_folder + '/' + file
                    tar.add(current_file, arcname='photo')
                elif file == "pickles":
                    current_file = self.data_root + image_folder + '/' + file
                    tar.add(current_file, arcname='pickles')
                else:
                    current_file = self.data_root + image_folder + '/' + file
                    tar.add(current_file, arcname='jsons')
            tar.close()
            return True
        except:
            return False

    def frame_preprocess_multiple(self, img, scales):
        """Preprocessing of obtained image (resizing, cropping, aligning, etc.)

        Parameters
        ----------
        img : np.ndarray
            Image read by opencv
        scales : str
            Default scales of the frame that is used

        Returns
        -------
        Aligned image
            head crop aligned horizontally having dimensions 112*112 pixels
        """
        thresh = self.detection_threshold
        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        faces, landmarks = self.detector.detect(img, thresh, scales=scales, do_flip=self.flip)

        aligned = []
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            # Getting the size of head rectangle
            height_y = box[3] - box[1]
            width_x = box[2] - box[0]
            # Calculating cropping area
            if landmarks is not None and height_y > self.min_head_size:
                center_y = box[1] + ((box[3] - box[1])/2)
                center_x = box[0] + ((box[2] - box[0])/2)
                rect_y = int(center_y - height_y/2)
                rect_x = int(center_x - width_x/2)
                landmark5 = landmarks[i].astype(np.int)
                aligned.append(align_img(img, landmark5))
        return aligned


    def get_vector_multiple(self, test_image):
        """Obtaining the embeddings from the given image

        Parameters
        ----------
        test_image : str
            Location of the image

        Returns
        -------
        feature
            embeddings (array) of size (n, 512)
        """
        try:
            if os.stat(test_image).st_size > 0:
                img = cv2.imread(test_image)
                crops = self.frame_preprocess_multiple(img, self.scales)
                features = []
                for crop in crops:
                    nimg = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    aligned = np.transpose(nimg, (2,0,1))
                    features.append(self.recognitor.get_feature(aligned))
                return features
            else:
                return None
        except:
            print('Bad Image')
            return None
