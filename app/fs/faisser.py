import psycopg2
import numpy as np
from shutil import copyfile
import faiss as fs
import os
import pickle

class Faisser:
    def __init__(self, update_pickles_dir, update_faiss_dir):
        self.update_pickles_dir = update_pickles_dir
        self.update_faiss_dir = update_faiss_dir

    
    def read_pickles(self, pickles_dir):
        # User glob to read recursively in subfolders
        data = None
        identificators = []
        vectors = []
        for pickle_file in os.listdir(pickles_dir):
            pickle_path = os.path.join(pickles_dir, pickle_file)
            with open(pickle_path,"rb") as f:
                try:
                    data = pickle.load(f)
                except EOFError:
                    return {'status': 'error', 'message': 'pickle not found in ' + pickle_path}
            for k in data.keys():
                identificators.append(k)
                vectors.append(data[k])

        # Formatting vectors and ids
        new_vectors = np.array(vectors, dtype=np.float32)
        new_ids = np.array(list(map(int, identificators)))
        
        return new_ids, new_vectors


    def get_records_amount(self, faiss_path):
        """Getting records amount stored in faiss index

        Returns
        -------
        amount : int
            amount of records in faiss index
        """
        if not os.path.exists(faiss_path):
            message = {
                    'status': 'error',
                    'message': 'NO FAISS FILE FOUND, PLEASE CHECK LOCATION OF INDEX'
                    }
        else:
            self.faiss_index = fs.read_index(faiss_path, fs.IO_FLAG_ONDISK_SAME_DIR)
        amount = self.faiss_index.ntotal
        return amount
    

    def create_block_and_index(self, new_vectors, new_ids, trained_index_path, path_to_save_new_block, merged_index_path, local_faiss_backup_path):
        """Create new block and new index file from vectors and ids

        Parameters
        ----------
        new_vectors : np.array
            (n,[1,512]) array with feature embeddings
        new_ids_np : np.array
            numpy integer array with ids
        trained_index_path : string
            path to read previously trained index
        path_to_save_new_block : string
            path to save newly created block
        merged_index_path : string
            path to save merged index
        local_faiss_backup_path : string
            path to save newly created index

        Returns
        -------
        result : boolean
            True or False
        """
        trained_index = fs.read_index(trained_index_path)
        trained_index.add_with_ids(new_vectors, new_ids)
		
		# Reading trained index and adding new vectors and ids to create new block
        blocks = len(os.listdir(path_to_save_new_block))
        fs.write_index(trained_index, path_to_save_new_block + "block_{}.index".format(str(blocks+1)))

		# Reading created blocks 1st block is created at the beginning
        ivfs = []
        for block in range(1, len(os.listdir(path_to_save_new_block))+1):
            # Reading from /final_index_backup/block_{}.index
            index = fs.read_index(path_to_save_new_block + "block_{}.index".format(str(block)), fs.IO_FLAG_MMAP)
            ivfs.append(index.invlists)
            index.own_invlists = False

        # Create final index
        index = fs.read_index(trained_index_path)
        # Saving /final_index_backup/merged_index.ivfdata
        invlists = fs.OnDiskInvertedLists(index.nlist, index.code_size, merged_index_path)
        ivf_vector = fs.InvertedListsPtrVector() 

        for ivf in ivfs: 
            ivf_vector.push_back(ivf)

        ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
        index.ntotal = ntotal
        index.replace_invlists(invlists)

        # Write final index
        try:
            fs.write_index(index, local_faiss_backup_path) # /final_index_backup/populated.index
            return {'status': True, "size": index.ntotal}
        except:
            return {'status': False, "size": None}