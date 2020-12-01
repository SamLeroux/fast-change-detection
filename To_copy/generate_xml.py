import xml.etree.cElementTree as ET
from random import randrange
import os
import sys
# sys.path.append('/home/hi-vision/')
# from Ugent.datasets.auto_annotate.scripts.detection_images import *


class GenerateXml(object):
    def __init__(self, box_array, im_width, im_height, inferred_class):
        self.inferred_class = inferred_class
        self.box_array = box_array
        self.im_width = im_width
        self.im_height = im_height

    def get_file_name(self):
        xml_path = '/home/hi-vision/Ugent/datasets/auto_annotate/image_output_xml/'
        directory = os.path.basename(xml_path)
        file_list = os.listdir(directory)
        print(file_list)
        # base = (os.path.basename(image_path))
        # global filename
        # filename = (os.path.splitext(base)[0])
        # print(filename)

        if len(file_list) == 0:
            return 1
        else:
            # return filename
            return len(file_list) + 1

    def gerenate_basic_structure(self,filename1):
        file_name = filename1
        print('generate_xml')
        print(filename1)

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = file_name + ".jpg"
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(self.im_width)
        ET.SubElement(size, "height").text = str(self.im_height)
        ET.SubElement(size, "depth").text = "3"

        for i in self.box_array:
            objectBox = ET.SubElement(annotation, "object")
            ET.SubElement(objectBox, "name").text = self.inferred_class
            # print(self.inferred_class)
            ET.SubElement(objectBox, "pose").text = "Unspecified"
            ET.SubElement(objectBox, "truncated").text = "0"
            ET.SubElement(objectBox, "difficult").text = "0"
            bndBox = ET.SubElement(objectBox, "bndbox")
            ET.SubElement(bndBox, "xmin").text = str(round(i['xmin']))
            ET.SubElement(bndBox, "ymin").text = str(round(i['ymin']))
            ET.SubElement(bndBox, "xmax").text = str(round(i['xmax']))
            ET.SubElement(bndBox, "ymax").text = str(round(i['ymax']))
        arquivo = ET.ElementTree(annotation)
        arquivo.write("/home/hi-vision/Ugent/datasets/auto_annotate/image_output_xml/" + file_name + ".xml")

def main():
    xml = GenerateXml([{'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}], '4000', '2000', 'miner') # just for debuggind
    xml.gerenate_basic_structure()
