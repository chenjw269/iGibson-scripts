import xml.dom.minidom


def read_area_xml(xml_pth):
    """
    Read area information from xml file

    Args:
        xml_pth (str): object xml file path

    Returns:
        list(list(int, int, int, int)): boundary of area
    """
    dom = xml.dom.minidom.parse(xml_pth)
    root = dom.documentElement

    xml_objects = root.getElementsByTagName('object')

    objects_p = []
    for xo in xml_objects:
        
        bbox = xo.getElementsByTagName('bndbox')[0]

        xmin = bbox.getElementsByTagName('xmin')[0].firstChild.data
        ymin = bbox.getElementsByTagName('ymin')[0].firstChild.data
        xmax = bbox.getElementsByTagName('xmax')[0].firstChild.data
        ymax = bbox.getElementsByTagName('ymax')[0].firstChild.data
        
        objects_p.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        
    return objects_p
