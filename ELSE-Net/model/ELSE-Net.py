from typing import List
import numpy as np

import open_clip
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F
from open_clip.transformer import VisionTransformer

from .clip_utils import (
    FeatureExtractor,
    FeatureExtractor_No,
    LearnableBgOvClassifier,
    PredefinedOvClassifier,
    RecWithAttnbiasHead,
    get_predefined_templates,
)
# from ..clipn.adapter import Adapter_Yes,Adapter_No
from .criterion import SetCriterion
from .matcher import HungarianMatcher
from .side_adapter import build_side_adapter_network
from ..clipn.open_clipn import get_clipn_classifier
from open_clipn.factory import create_model_and_transforms
@META_ARCH_REGISTRY.register()
class SAN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_visual_extractor: nn.Module,
        clip_rec_head: nn.Module,
        # adapter_yes: Adapter_Yes,
        # adapter_no: Adapter_No,
        # clipn_visual_extractor:nn.Module,
        side_adapter_network: nn.Module,
        ov_classifier: PredefinedOvClassifier,
        criterion: SetCriterion,
        size_divisibility: int,
        asymetric_input: bool = True,
        clip_resolution: float = 0.5,
        pixel_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        pixel_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        sem_seg_postprocess_before_inference: bool = False,
    ):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility
        self.criterion = criterion
        # self.mutual_loss_weight = 1
        self.side_adapter_network = side_adapter_network
        self.clip_visual_extractor = clip_visual_extractor
        # self.clipn_visual_extractor = clipn_visual_extractor
        # self.adapter_yes = adapter_yes
        # self.adapter_no = adapter_no
        # if isinstance(self.adapter, nn.Module):
        #     self.adapter.train()
        # else:
        #     raise TypeError("adapter must be an instance of nn.Module")
        # adapter_yes.train()
        # adapter_no.train()
        self.clip_rec_head = clip_rec_head
        # self.clipn_encoder = VisualTransformer
        self.ov_classifier = ov_classifier
        # pre_train = '/root/autodl-tmp/SAN-main/san/model/CLIPN_ATD_Repeat2_epoch_10.pt'
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clipn_model, process_train, process_test = create_model_and_transforms('ViT-B/16', pretrained = pre_train, device=device,freeze=True)
        # self.clipn_model.eval()
        # for param in self.clipn_model.parameters():
        #     param.requires_grad = False  
        # self.weight_yes_1, self.weight_no_2 = get_clipn_classifier('coco_2017_train_stuff_sem_seg', self.clipn_model, self.clip_visual_extractor, device=device)
        # self.weight_yes_3, self.weight_no_4 = get_clipn_classifier('voc_sem_seg_val', self.clipn_model, self.clip_visual_extractor, device=device)
        # self.weight_yes_5, self.weight_no_6 = get_clipn_classifier('pcontext_sem_seg_val', self.clipn_model, self.clip_visual_extractor, device=device)
        # self.weight_yes_7, self.weight_no_8 = get_clipn_classifier('ade20k_sem_seg_val', self.clipn_model, self.clip_visual_extractor, device=device)
        # self.weight_yes_9, self.weight_no_10 = get_clipn_classifier('ade20k_full_sem_seg_val', self.clipn_model, self.clip_visual_extractor, device=device)
        # self.clipn_classifier_1 = get_clipn_classifier('coco_2017_train_stuff_sem_seg', self.clipn_model,device=device)
        # self.clipn_classifier_2 = get_clipn_classifier('voc_sem_seg_val', self.clipn_model, device=device)
        # self.clipn_classifier_3 = get_clipn_classifier('pcontext_sem_seg_val', self.clipn_model, device=device)
        # self.clipn_classifier_4 = get_clipn_classifier('ade20k_sem_seg_val', self.clipn_model,device=device)
        # self.clipn_classifier_5 = get_clipn_classifier('ade20k_full_sem_seg_val', self.clipn_model,device=device)
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        ## copied from maskformer2
        # Loss parameters
        no_object_weight = cfg.MODEL.SAN.NO_OBJECT_WEIGHT
        # loss weights
        class_weight = cfg.MODEL.SAN.CLASS_WEIGHT
        dice_weight = cfg.MODEL.SAN.DICE_WEIGHT
        mask_weight = cfg.MODEL.SAN.MASK_WEIGHT
        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }
        aux_weight_dict = {}
        for i in range(len(cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS) - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks"]

        criterion = SetCriterion(
            num_classes=cfg.MODEL.SAN.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.SAN.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.SAN.IMPORTANCE_SAMPLE_RATIO,
        )
        ## end of copy

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.MODEL.SAN.CLIP_MODEL_NAME,
            pretrained=cfg.MODEL.SAN.CLIP_PRETRAINED_NAME,
        )
        # adapter_yes = Adapter_Yes()
        # adapter_no = Adapter_No()
        ov_classifier = LearnableBgOvClassifier(
            model, templates=get_predefined_templates(cfg.MODEL.SAN.CLIP_TEMPLATE_SET)
        )
        clip_visual_extractor = FeatureExtractor(
            model.visual,
            last_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
            frozen_exclude=cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE,
        )
        clipn_visual_extractor = FeatureExtractor_No(
            model.visual,
            last_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
            frozen_exclude=cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE,
        )
        clip_rec_head = RecWithAttnbiasHead(
            model.visual,
            first_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
            frozen_exclude=cfg.MODEL.SAN.CLIP_DEEPER_FROZEN_EXCLUDE,
            cross_attn=cfg.MODEL.SAN.REC_CROSS_ATTN,
            sos_token_format=cfg.MODEL.SAN.SOS_TOKEN_FORMAT,
            sos_token_num=cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES,
            downsample_method=cfg.MODEL.SAN.REC_DOWNSAMPLE_METHOD,
        )
        pixel_mean, pixel_std = (
            preprocess.transforms[-1].mean,
            preprocess.transforms[-1].std,
        )
        pixel_mean = [255.0 * x for x in pixel_mean]
        pixel_std = [255.0 * x for x in pixel_std]

        return {
            "clip_visual_extractor": clip_visual_extractor,
            # "clipn_visual_extractor": clipn_visual_extractor,
            "clip_rec_head": clip_rec_head,
            "side_adapter_network": build_side_adapter_network(
                cfg, clip_visual_extractor.output_shapes
            ),
            "ov_classifier": ov_classifier,
            "criterion": criterion,
            "size_divisibility": cfg.MODEL.SAN.SIZE_DIVISIBILITY,
            "asymetric_input": cfg.MODEL.SAN.ASYMETRIC_INPUT,
            "clip_resolution": cfg.MODEL.SAN.CLIP_RESOLUTION,
            "sem_seg_postprocess_before_inference": cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
        }

    def forward(self, batched_inputs):
        # get classifier weight for each dataset
        # !! Could be computed once and saved. It will run only once per dataset.
        if "vocabulary" in batched_inputs[0]:
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_vocabulary(
                    batched_inputs[0]["vocabulary"]
                )
            )
        else:
            dataset_names = [x["meta"]["dataset_name"] for x in batched_inputs]
            assert (
                len(list(set(dataset_names))) == 1
            ), "All images in a batch must be from the same dataset."
            ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_dataset_name(dataset_names[0])
            )  # C+1,ndim
        # if 'coco_2017_train_stuff_sem_seg' in dataset_names[0]:
        #     clipn_classifier_weight = self.clipn_classifier_1
        # if 'voc_sem_seg_val' in dataset_names[0]:
        #     clipn_classifier_weight = self.clipn_classifier_2
        # if 'pcontext_sem_seg_val' in dataset_names[0]:
        #     clipn_classifier_weight = self.clipn_classifier_3
        # if 'ade20k_sem_seg_val' in dataset_names[0]:
        #     clipn_classifier_weight = self.clipn_classifier_4
        # if 'ade20k_full_sem_seg_val' in dataset_names[0]:
        #     clipn_classifier_weight = self.clipn_classifier_5  
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        clip_input = images.tensor
        if self.asymetric_input:
            clip_input = F.interpolate(
                clip_input, scale_factor=self.clip_resolution, mode="bilinear"
            )
        clip_image_features = self.clip_visual_extractor(clip_input)

        # clip_image_features_adapter = self.clipn_visual_extractor(clip_input)
        # fusion_image_features = (clip_image_features+clip_image_features_adapter)
        # clipn_classifier_weight = clipn_classifier(clip_input)
        # probabilities_1 = torch.softmax(ov_classifier_weight,dim=0)
        # probabilities_2 = torch.softmax(clipn_classifier_weight,dim=0)
        # probabilities_1 = probabilities_1[:,1]
        # probabilities_2 = probabilities_2[:,1]
        # top5_prob, top5_catid = torch.topk(probabilities_2, 5)
        # top5_prob_1, top5_catid_1 = torch.topk(probabilities_1, 5)
        # probabilities_1_index = probabilities_1[top5_catid]
        mask_preds, attn_biases = self.side_adapter_network(
            images.tensor, clip_image_features
        )
        # mask_preds_no, attn_biases = self.side_adapter_network(
        #     images.tensor, clip_image_features
        # )
        # !! Could be optimized to run in parallel.
        # fusion_weight = ov_classifier_weight - clipn_classifier_weight 
        # fusion_weight =  ov_classifier_weight * clipn_classifier_weight
        mask_embs_yes = [
            self.clip_rec_head(clip_image_features, attn_bias, normalize=True)
            for attn_bias in attn_biases
        ]  # [B,N,C]
        mask_logits = [
            torch.einsum("bqc,nc->bqn", mask_emb, ov_classifier_weight)
            for mask_emb in mask_embs_yes
        ]
        # mask_embs_no = [
        #     self.clip_rec_head(fusion_image_features, attn_bias, normalize=True)
        #     for attn_bias in attn_biases
        # ]  # [B,N,C]
        # mask_logits_no = [
        #     torch.einsum("bqc,nc->bqn", mask_emb, clipn_classifier_weight)
        #     for mask_emb in mask_embs_no
        # ]
        # mask_logits_clipn = [
        #     torch.einsum("bqc,nc->bqn", mask_emb, clipn_classifier_weight)
        #     for mask_emb in mask_embs
        # ]
        # mask_logits_no = 20*mask_logits_no
        # intersection=[]
        # indices_1=[]
        # for i in range(len(mask_logits_yes)):
        #     probabilities_2 = F.softmax(mask_logits_no[i],dim=-1)
            # probabilities_pro = probabilities_2[2]
            # max_values, max_indices = torch.max(probabilities_pro)

            # probabilities_2_yes = 1-probabilities_2
            # top5_prob, top5_catid_no = torch.topk(probabilities_2_yes, 5)
            # threshold = 0.2
            # result = (probabilities_2 > threshold).nonzero(as_tuple=False)
            # indices_1.append(result)
            # top5_prob_yes, top5_catid_yes = torch.topk(probabilities_1, 5)
            # top5_catid_yes = top5_catid_yes.cpu()
            # top5_catid_no = top5_catid_no.cpu()
        # mask_logits = []
        # for i in range(len(mask_logits_yes)):
        #     mask_logits_tensor = mask_logits_yes[i]
        #     mask_no_logits_tensor = mask_logits_no[i]
        #     result = mask_logits_tensor - mask_no_logits_tensor
        #     mask_logits.append(result)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # total_correct = 0
            # total_samples = 0 
            # for i in range(len(targets)):
            #     tensor_values = targets[i]['labels']
            #     correct_count = torch.sum(torch.isin(intersection,tensor_values)).item()
            #     total_correct += correct_count
            #     total_samples += len(tensor_values)
            # accuracy = total_correct / total_samples
            outputs = {
                "pred_logits": mask_logits[-1],
                "pred_masks": mask_preds[-1],
                "ov_classifier_weight": ov_classifier_weight,  
                "aux_outputs": [
                    {
                        "pred_logits": aux_pred_logits,
                        "pred_masks": aux_pred_masks,
                    }
                    for aux_pred_logits, aux_pred_masks in zip(
                        mask_logits[:-1], mask_preds[:-1]
                    )
                ],
            }
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
            
        else:
            mask_preds = mask_preds[-1]
            mask_logits = mask_logits[-1]
            # torch.cuda.empty_cache()
            # Inference
            mask_preds = F.interpolate(
                mask_preds,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_logits, mask_preds, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    @property
    def device(self):
        return self.pixel_mean.device

