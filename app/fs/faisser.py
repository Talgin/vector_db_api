import psycopg2
import numpy as np
from shutil import copyfile
import faiss as fs
from faiss.contrib.ondisk import merge_ondisk
import os
from glob import glob
import pickle

class Faisser:
    def __init__(self, updated_pickles_dir, updated_faiss_dir):
        self.updated_pickles_dir = updated_pickles_dir
        self.updated_faiss_dir = updated_faiss_dir

    
    def read_ids_from_postgres_db(self, pg_server, pg_port, pg_db, pg_user, pg_pass, pg_schema_and_table):
        try:
            # connect to the PostgreSQL database
            conn = psycopg2.connect(host=pg_server, port=pg_port, dbname=pg_db, user=pg_user, password=pg_pass)
            # create a new cursor object
            cur = conn.cursor()
            # execute the INSERT statement
            sql_query = """SELECT ud_code 
                        FROM fr.unique_ud_gr"""
            cur.execute(sql_query)
            # commit the changes to the database
            blob = cur.fetchall()
            # close the communication with the PostgreSQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return None
        finally:
            if conn is not None:
                conn.close()
        return blob

    
    def read_pickles(self):
        # User glob to read recursively in subfolders
        data = None
        identificators = []
        vectors = []
        for pickle_file in glob(self.updated_pickles_dir + '/**/*.pickle'):
            # pickle_path = os.path.join(root, pickle_file)
            with open(pickle_file,"rb") as f:
                try:
                    data = pickle.load(f)
                except EOFError:
                    return {'status': 'error', 'message': 'pickle not found in ' + pickle_file}
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
    

    def create_block_and_index(self, new_ids, new_vectors, trained_index_path, new_faiss_index_dir):
        """Create new block and new index file from vectors and ids

        Parameters
        ----------
        new_vectors : np.array
            (n,[1,512]) array with feature embeddings
        new_ids_np : np.array
            numpy integer array with ids
        dir_with_pickles:
            name of the directory of todays pickles
        trained_index_path : string
            path to read previously trained index
        new_faiss_index_dir : string
            path to save merged and populated index files

        Returns
        -------
        result : boolean
            True or False
        """
        index = fs.read_index(trained_index_path)
        index.add_with_ids(new_vectors, new_ids)
		# Reading trained index and adding new vectors and ids to create new block
        fs.write_index(index, os.path.join(new_faiss_index_dir, "block.index"))

		# Reading created blocks 1st block is created at the beginning
        ivfs = []
        # Reading from /updated_final_index/block.index
        index = fs.read_index(os.path.join(new_faiss_index_dir, "block.index"), fs.IO_FLAG_MMAP)
        ivfs.append(index.invlists)
        index.own_invlists = False

        # Reading trained index to create final index
        index = fs.read_index(trained_index_path)
        # Creating merged index from inverted lists (blocks): /updated_final_index/merged_index.ivfdata
        invlists = fs.OnDiskInvertedLists(index.nlist, index.code_size, os.path.join(new_faiss_index_dir, "merged_index.ivfdata"))
        ivf_vector = fs.InvertedListsPtrVector()

        for ivf in ivfs:
            ivf_vector.push_back(ivf)

        ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
        index.ntotal = ntotal
        index.replace_invlists(invlists)

        # Saving final index in /updated_final_index/populated.index
        try:
            fs.write_index(index, os.path.join(new_faiss_index_dir, 'populated.index'))
            return {'status': True, "size": index.ntotal, 'path': os.path.join(new_faiss_index_dir, 'populated.index')}
        except:
            return {'status': False, "size": None}