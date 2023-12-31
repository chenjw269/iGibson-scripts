import xml.dom.minidom


def read_wall_xml(xml_pth):
    """Read wall mask from xml file

    Args:
        xml_pth (str): wall xml file path

    Returns:
        list(list): wall bounding box
    """
    dom = xml.dom.minidom.parse(xml_pth)
    root = dom.documentElement

    xml_objects = root.getElementsByTagName('object')
    # print(f"{len(xml_objects)} objects")

    objects_p = []
    for xo in xml_objects:
        
        bbox = xo.getElementsByTagName('bndbox')[0]

        xmin = bbox.getElementsByTagName('xmin')[0].firstChild.data
        ymin = bbox.getElementsByTagName('ymin')[0].firstChild.data
        xmax = bbox.getElementsByTagName('xmax')[0].firstChild.data
        ymax = bbox.getElementsByTagName('ymax')[0].firstChild.data
        
        objects_p.append((int(xmin), int(ymin), int(xmax), int(ymax)))
        
    return objects_p


import json


def read_obj_json(json_pth):
    """
    Read object information from json file

    Args:
        json_pth (str): object json file path

    Returns:
        list(list(str, float, float)): semantic label and position of objects
    """
    
    obj_list = []
    
    with open(json_pth, encoding="utf-8") as f:
        
        js_data = json.load(f)
        
        for item in js_data:
            
            # object semantic label
            k1 = "category"
            v1 = item[k1]
            label = v1
    
            # object X, Y axis center
            k2 = "center"
            v2 = (item[k2][0], item[k2][1])
            x_center = round(v2[0], 4)
            y_center = round(v2[1], 4)
            
            # object Z axis center and scale
            k3 = "z"
            v3 = (item[k3][0], item[k3][1])
            z_center = round((v3[0] + v3[1]) / 2, 4)
            z_scale = round(abs(v3[0] - v3[1]), 4)
            
            # object X, Y axis scale
            k4 = "raw_pts"
            v4 = item[k4]
            
            x_list = [p[0] for p in v4]
            y_list = [p[1] for p in v4]

            x_max = max(x_list)
            x_min = min(x_list)
            y_max = max(y_list)
            y_min = min(y_list)
            x_scale = round(x_max - x_min, 4)
            y_scale = round(y_max - y_min, 4)
            
            obj_list.append([label, x_center, y_center, x_scale, y_scale, z_center, z_scale])


    return obj_list


class Vocab:
    """
    Class of vocabulary
    """
    def __init__(self):
        
        self.word_to_index = {}
        self.index_to_word = []
        
    def add(self, token):
        """Add a word to vocabulary

        Args:
            token (str): the word to add

        Returns:
            int: the word index in vocabulary
        """
        token = token.lower()
        if token in self.index_to_word:
            pass
        else:
            self.index_to_word.append(token)
            self.word_to_index[token] = len(self.index_to_word)
            
        return self.word_to_index[token]


import numpy as np


def generate_random_colors(num_colors=512):
    """Generate random RGB colors for the specified number of colors

    Args:
        num_colors (int, optional): colors number. Defaults to 512.

    Returns:
        list(list): color RGB value
    """

    back_ground = np.expand_dims(np.array([0, 0, 0]), axis=0)
    colors = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8) / 255
    
    colors = np.concatenate((back_ground, colors), axis=0)
    
    return colors