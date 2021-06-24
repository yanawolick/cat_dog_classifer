class find_pet():
    
    def __init__(self, modl_way = '/home/polysh/Project_Find_cat/mask_rcnn_coco.h5'):  
        self.segment_img = instance_segmentation()
        self.segment_img.load_model(modl_way)
        self.cat_dog_class = self.segment_img.select_target_classes(cat = True, dog = True)
        
    
    def pet_detection(self, img_way):
        detector = self.segment_img.segmentImage(image_path = img_way,
                                 segment_target_classes = self.cat_dog_class)
        return detector
        
    def cat_or_dog(self, img_way):
        detector = self.segment_img.segmentImage(image_path = img_way,
                                 segment_target_classes = self.cat_dog_class)
        kind = detector[0]['class_ids']
        if kind is None or len(kind) > 1 or len(kind) == 0:
            return 0
        else:
            if kind[0] == 16:
                return 'cat'
            else:
                return 'dog'

        


from pixellib.instance import instance_segmentation 