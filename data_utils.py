import pandas as pd
import os
from tqdm.notebook import tqdm

def read_uuid(rootdir, cancer_type, uuid):
    """Reads a case file into DataFrame, appends UUID and cancer type as column
    """
    try:
        return (
            pd
            .read_csv(os.path.join("..", "data", cancer_type, uuid, 
                                    next(filter( lambda s : s.endswith("tsv"),os.listdir(os.path.join(rootdir, "data", cancer_type, uuid)) ))
                                   ),  sep="\t", skiprows=[0,2,3,4,5])
            [['gene_name', "gene_type", 'unstranded', 'stranded_first', 'stranded_second', 'tpm_unstranded','fpkm_unstranded','fpkm_uq_unstranded']]
            .assign(uuid = uuid, cancer_type = cancer_type)
        )
    except Exception as e:
        print(f"Exception occured for {uuid}\n{e}")
        
def load_all_cases(cancer_type, gene_type):
    uuids = os.listdir(os.path.join("..", "data", cancer_type))
    df = (
        pd
        .concat([read_uuid("..", cancer_type, uuid) for uuid in tqdm(uuids)])
        .query(f"gene_type=='{gene_type}'")
        .pivot_table(
          index=['cancer_type','uuid'], 
          columns = 'gene_name', 
          values= ['tpm_unstranded'] # ['unstranded','stranded_first','stranded_second','tpm_unstranded','fpkm_unstranded','fpkm_uq_unstranded'])
        )
        .reset_index()
    )
    df.columns = [col1 if col1!='tpm_unstranded' else col2 for col1,col2 in df.columns]
    return df

def add_dataset_col(df, uuid_train,uuid_val,uuid_test):
    
    df=df.copy()
    df.loc[df.uuid.isin(uuid_train), 'dataset'] = 'Train'
    df.loc[df.uuid.isin(uuid_val), 'dataset'] = 'Validation'
    df.loc[df.uuid.isin(uuid_test), 'dataset'] = 'Test'
    return df

def cr_gene_type_df(gene_type, cancer_types):
    df = pd.concat([load_all_cases(cancer_type, gene_type) for cancer_type in cancer_types])    
    return df