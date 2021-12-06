# %%
"""
Example of a chronjob
"""

# %%
import pandas as pd
import tree_maker
from tree_maker import NodeJob

def load_tree(filename):
    try:
        root=tree_maker.tree_from_json(filename)
        return root
    except Exception as e:
        print(e)
        print('Probably you forgot to edit the address of you json file...')
    
def get_info(root):
    my_list = []
    for node in root.descendants:
        my_dict = tree_maker.from_json(node.log_file)
        my_dict['parent'] = node.parent
        my_dict['name'] = node.name 
        my_dict['path'] = node.path
        my_list.append(my_dict)        
    return pd.DataFrame(my_list)

def get_list_descendant(root, operation='completed'):
    for node in root.descendants:
        if node.has_not_been(operation):
            print(node.path)

# %%
# Load the tree from a yaml
if __name__=='__main__':
    root = load_tree('./study_001/tree.json')
    if root.has_been('completed'):
        print('All descendants of root are completed!')
    else:
        for node in root.descendants:
            node.smart_run()
        if all([descendant.has_been('completed') for descendant in root.descendants]):
            root.tag_as('completed')
            print('All descendants of root are completed!')
        else:
            for descendant in root.descendants:
                if descendant.has_not_been('completed'):
                     print(descendant.path)

