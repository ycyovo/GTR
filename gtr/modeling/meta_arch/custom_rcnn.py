# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/rcnn.py
# The original file is under Apache-2.0 License

# Modified by Xingyi Zhou: support not_clamp_box

from typing import Dict, List, Optional, Tuple
import torch
import cv2
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from gtr.data.custom_build_augmentation import build_custom_augmentation

from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.data import transforms as T

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import Instances, ROIMasks, Boxes

from yolox.utils import (
    postprocess,
    xyxy2xywh,
    fuse_model
)

def custom_detector_postprocess(
    results: Instances, output_height: int, output_width: int, 
    mask_threshold: float = 0.5, not_clamp_box=False,
):
    """
    allow not clamp box for MOT datasets
    """
    # Change to 'if is_tracing' after PT1.7
    if isinstance(output_height, torch.Tensor):
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    if not not_clamp_box:
        output_boxes.clip(results.image_size) # TODO (Xingyi): note modified

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results

def get_model():
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    # if getattr("model", None) is None:
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(4/3, 1.25, in_channels=in_channels)
    head = YOLOXHead(1, 1.25, in_channels=in_channels)
    head.training = False
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    model.training = False

    model.cuda(0)
    model.eval()
    ckpt = torch.load( 'pretrained/bytetrack_ablation.pth.tar' , 'cuda:0' )
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    model = model.eval()
    model = model.half()

    return model

def preproc(image, input_size=(800,1440), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    # padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

def convert_to_coco_format(outputs, info_imgs, shape_target):
    list_instances = []
    img_h_target = shape_target[1]
    img_w_target = shape_target[2]
    for (output, img_h_ori, img_w_ori ) in zip(
        outputs, info_imgs[0], info_imgs[1]
    ):
        if output is None:
            continue
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            800 / float(img_h_ori), 1440 / float(img_w_ori)
        )
        bboxes /= scale
        scale1 = img_w_target / img_w_ori
        bboxes *= scale1 
        scores = output[:, 4] * output[:, 5]
        # bboxes = xyxy2xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            # label = self.dataloader.dataset.class_ids[int(cls[ind])]
            this_instance = Instances( (img_h_target, img_w_target) )
            this_instance.set('scores', torch.tensor([scores[ind].numpy().item()]))
            this_instance.set('pred_classes' , torch.tensor([0]))
            this_instance.set('proposal_boxes',  Boxes( bboxes[ind].unsqueeze(0)  ))
            this_instance.set('objectness_logits', torch.tensor([scores[ind].numpy().item()]))
            list_instances.append(this_instance.to('cuda:0'))
            # pred_data = {
            #     "bbox": bboxes[ind].numpy().tolist(),
            #     "score": scores[ind].numpy().item(),
            #     "segmentation": [],
            # }  # COCO json format
    return Instances.cat(list_instances)

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Allow not clip box for MOT datasets
    '''
    @configurable
    def __init__(
        self, **kwargs):
        """
        add not_clamp_box
        """
        not_clamp_box = kwargs.pop('not_clamp_box', False)
        super().__init__(**kwargs)
        self.not_clamp_box = not_clamp_box
        self.yolo_model = get_model()

    @classmethod
    def from_config(cls, cfg):  
        ret = super().from_config(cfg)
        ret['not_clamp_box'] = cfg.INPUT.NOT_CLAMP_BOX
        return ret

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        cfg,
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Allow not clamp box for MOT datasets
        """
        image_tensor = batched_inputs[0]['image']
        image_numpy = image_tensor.numpy()

        aug_input = T.StandardAugInput(image_numpy)
        augmentations = build_custom_augmentation(cfg,False)
        transforms = aug_input.apply_augmentations(augmentations)
        image_features = aug_input.image
        image_features = torch.as_tensor(np.ascontiguousarray(image_features.transpose(2, 0, 1)))
        
        # dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        batched_inputs[0]['image'] = image_features
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert not self.training
        tensor_type = torch.cuda.HalfTensor

        image_numpy = image_numpy[:, :, ::-1]
        image_after_preproc = preproc(image_numpy)
        
        # image -> format for yolo
        image_for_yolo = torch.from_numpy(np.asarray(image_after_preproc).copy()).clone() #.to('cuda:0')
        image_for_yolo = image_for_yolo.unsqueeze(0)
        image_for_yolo = image_for_yolo.float()
        image_for_yolo = image_for_yolo.type(tensor_type)

        outputs = self.yolo_model( image_for_yolo )

        # import ipdb; ipdb.set_trace()
        outputs = postprocess(outputs, 1, 0.01, 0.7)
        info_imgs = [ torch.tensor([(image_tensor.shape)[0]]) , torch.tensor([(image_tensor.shape)[1]]) ]
        output_results = convert_to_coco_format(outputs, info_imgs , images[0].shape)
        
        results, _ = self.roi_heads(images, features, [output_results], None)
        return results

        # if detected_instances is None:
        #     if self.proposal_generator is not None:
        #         proposals, _ = self.proposal_generator(images, features, None)
        #     else:
        #         assert "proposals" in batched_inputs[0]
        #         proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        #     results, _ = self.roi_heads(images, features, proposals, None)
        # else:
        #     detected_instances = [x.to(self.device) for x in detected_instances]
        #     results = self.roi_heads.forward_with_given_boxes(
        #         features, detected_instances)

        # if do_postprocess:
        #     assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
        #     return CustomRCNN._postprocess(
        #         results, batched_inputs, images.image_sizes,
        #         not_clamp_box=self.not_clamp_box)
        # else:
        #     return results

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], 
        image_sizes, not_clamp_box=False):
        """
        Allow not clip box for MOT datasets
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = custom_detector_postprocess(
                results_per_image, height, width, not_clamp_box=not_clamp_box)
            processed_results.append({"instances": r})
        return processed_results