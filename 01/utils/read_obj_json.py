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
        print(f"JSON file consists {len(js_data)} objects")
        
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