# make a function to load the information of the jupyter notebook (cell number and the information)
def load_jupyter(path:str) -> str:
    '''  
    Loads the jupyter notebook information in string format\n\n

    INPUTS:\n
    path: Provide the path where the file is found \n\n

    OUTPUTS:\n
    whole_info: returns the information in STRING format about the cell number with the respective information\n\n
    '''
    # import the library
    import json

    # load the jupyter notebook 
    with open(path, encoding='utf-8') as jupyter: 
        file = json.load(jupyter) # load the json file from the jupyter cells

    whole_info = ""
    for number_cell, info in enumerate(file['cells']):
        whole_info+=f"Number Cell: {number_cell + 1}\n\n {info['source']}\n\n"

    return whole_info



#print(load_jupyter('azure.ipynb'))

