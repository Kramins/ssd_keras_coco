from pycocotools.coco import COCO
import numpy as np

class CocoDataGenerator:
    def __init__(self, categoryies, annotation_file,coco_data_dir):
        self.categories = categoryies
        self.annotation_file = annotation_file
        self.coco_data_dir = coco_data_dir

        self.coco_image_dir = self.coco_data_dir + 'images/'

    def get_data(self):
        coco =COCO(annotation_file=self.annotation_file)
        cocoCategorys_ids = coco.getCatIds(catNms=self.categories);
        imgIds = set()

        for cat_id in cocoCategorys_ids:

            cat_img_ids = coco.getImgIds(catIds=cat_id)
            for i in cat_img_ids:
                imgIds.add(i)

        num_classes = len(cocoCategorys_ids)
        data = dict()
        imgIds = list(imgIds)[0:2000]
        imgs = coco.loadImgs(imgIds)

        coco.download(self.coco_image_dir,imgIds)

        for img in imgs:
            imgId = img['id']
            annIds = coco.getAnnIds(imgIds=imgId, catIds=cocoCategorys_ids, iscrowd=None)
            anns = coco.loadAnns(annIds)

            image_width = img['width']
            image_height = img['height']
            bounding_boxes = []
            one_hot_classes = []

            for ann in anns:
                one_hot_vector = [0] * num_classes

                cat_index = cocoCategorys_ids.index(ann['category_id'])

                bbox = ann['bbox']
                xmin = float(bbox[0]) / image_width
                ymin = float(bbox[1]) / image_height
                xmax = float(bbox[2]) / image_width
                ymax = float(bbox[3]) / image_height

                bounding_box = [xmin, ymin, xmax, ymax]
                one_hot_vector[cat_index] = 1

                bounding_boxes.append(bounding_box)
                one_hot_classes.append(one_hot_vector)

            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            dict_key = img['file_name']
            data[dict_key] = image_data

        return  self.categories, data