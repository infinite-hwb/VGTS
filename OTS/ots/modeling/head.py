import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_coder import OTSBoxCoder, BoxGridGenerator
from ots.structures.feature_map import FeatureMapSize
from ots.structures.bounding_box import BoxList, cat_boxlist


def build_OTS_head_creator(do_simple_affine, is_cuda, use_inverse_geom_model, feature_map_stride, feature_map_receptive_field):
    aligner = OTSAlignment(do_simple_affine, is_cuda, use_inverse_geom_model)
    head_creator = OTSHeadCreator(aligner, feature_map_stride, feature_map_receptive_field)
    return head_creator


def convert_box_coordinates_local_to_global(resampling_grids, default_boxes_xyxy):
    box_transforms_x_A = (default_boxes_xyxy.narrow(-1, 2, 1) - default_boxes_xyxy.narrow(-1, 0, 1)) / 2
    box_transforms_x_B = (default_boxes_xyxy.narrow(-1, 2, 1) + default_boxes_xyxy.narrow(-1, 0, 1)) / 2
    box_transforms_y_A = (default_boxes_xyxy.narrow(-1, 3, 1) - default_boxes_xyxy.narrow(-1, 1, 1)) / 2
    box_transforms_y_B = (default_boxes_xyxy.narrow(-1, 3, 1) + default_boxes_xyxy.narrow(-1, 1, 1)) / 2

    resampling_grids_size = [-1] * resampling_grids.dim()
    resampling_grids_size[-2] = resampling_grids.size(-2)
    resampling_grids_size[-3] = resampling_grids.size(-3)
    add_dims = lambda x: x.unsqueeze(-2).unsqueeze(-3).expand(resampling_grids_size)
    b_x_A = add_dims(box_transforms_x_A)
    b_x_B = add_dims(box_transforms_x_B)
    b_y_A = add_dims(box_transforms_y_A)
    b_y_B = add_dims(box_transforms_y_B)
    resampling_grids_x = resampling_grids.narrow(-1, 0, 1) * b_x_A + b_x_B
    resampling_grids_y = resampling_grids.narrow(-1, 1, 1) * b_y_A + b_y_B
    resampling_grids_global = torch.cat([resampling_grids_x, resampling_grids_y], -1)

    return resampling_grids_global


class OTSAlignment(nn.Module):
    def __init__(self, do_simple_affine, is_cuda, use_inverse_geom_model):
        super(OTSAlignment, self).__init__()

        self.model_type = "affine" if not do_simple_affine else "simple_affine"
        self.use_inverse_geom_model = use_inverse_geom_model
        if self.model_type == "affine":
            transform_net_output_dim = 6
        elif self.model_type == "simple_affine":
            transform_net_output_dim = 4
        else:
            raise(RuntimeError("Unknown transformation model \"{0}\"".format(self.model_type)))
        self.out_grid_size = FeatureMapSize(w=15, h=15)
        self.reference_feature_map_size = FeatureMapSize(w=15, h=15)
        self.network_stride = FeatureMapSize(w=1, h=1)
        self.network_receptive_field = FeatureMapSize(w=15, h=15)

        self.input_feature_dim = self.reference_feature_map_size.w * self.reference_feature_map_size.h
        self.parameter_regressor = TransformationNet(output_dim=transform_net_output_dim,
                                                     use_cuda=is_cuda,
                                                     normalization='batchnorm',
                                                     kernel_sizes=[7, 5],
                                                     channels=[128, 64],
                                                     input_feature_dim=self.input_feature_dim)

    def prepare_transform_parameters_for_grid_sampler(self, transform_parameters):
        num_params = transform_parameters.size(1)
        transform_parameters = transform_parameters.transpose(0, 1).contiguous().view(num_params, -1)
    
        if self.model_type == "affine":
            assert num_params == 6, f'Affine transformation parameter vector has to be of dimension 6, have {num_params} instead'
            transform_parameters = transform_parameters.transpose(0, 1).view(-1, 2, 3)
        elif self.model_type == "simple_affine":
            assert num_params == 4, f'Simplified affine transformation parameter vector has to be of dimension 4, have {num_params} instead'
            zeros_to_fill_blanks = torch.zeros_like(transform_parameters[0])
            transform_parameters = torch.stack([transform_parameters[0], zeros_to_fill_blanks, transform_parameters[1],
                                                zeros_to_fill_blanks, transform_parameters[2], transform_parameters[3]], dim=1)
            transform_parameters = transform_parameters.view(-1, 2, 3)
        else:
            raise RuntimeError(f"Unknown transformation model \"{self.model_type}\"")
    
        if self.use_inverse_geom_model:
            assert self.model_type in ["affine", "simple_affine"], "Inversion of the transformation is implemented only for the affine transformations"
            assert transform_parameters.size(-2) == 2 and transform_parameters.size(-1) == 3, f"transform_parameters should be of size ? x 2 x 3 to interpret them as affine matrix, have {transform_parameters.size()} instead"
            grid_batch_size = transform_parameters.size(0)
            lower_row = torch.zeros(grid_batch_size, 1, 3, device=transform_parameters.device, dtype=transform_parameters.dtype)
            lower_row[:, :, 2] = 1
    
            full_matrices = torch.cat([transform_parameters, lower_row], dim=1)
    
            def robust_inverse(batchedTensor):
                try:
                    inv = torch.inverse(batchedTensor)
                except:
                    n = batchedTensor.size(1)
                    batchedTensor_reg = batchedTensor.clone().contiguous()
                    for i in range(n):
                        batchedTensor_reg[:, i, i] = batchedTensor_reg[:, i, i] + 1e-5
                    inv = torch.inverse(batchedTensor_reg)
                return inv
    
            def batched_inverse(batchedTensor):
                if batchedTensor.shape[0] >= 256 * 256 - 1:
                    temp = []
                    for t in torch.split(batchedTensor, 256 * 256 - 1):
                        temp.append(robust_inverse(t))
                    return torch.cat(temp)
                else:
                    return robust_inverse(batchedTensor)
    
            inverted = batched_inverse(full_matrices)
            transform_parameters = inverted[:, :2, :]
            transform_parameters = transform_parameters.contiguous()
        return transform_parameters


    def forward(self, corr_maps):
        batch_size, fm_height, fm_width = corr_maps.size(0), corr_maps.size(-2), corr_maps.size(-1)
        assert corr_maps.size(1) == self.input_feature_dim, f"The dimension 1 of corr_maps={corr_maps.size(1)} should be equal to self.input_feature_dim={self.input_feature_dim}"
        
        transform_parameters = self.parameter_regressor(corr_maps)
        transform_parameters = self.prepare_transform_parameters_for_grid_sampler(transform_parameters)
    
        resampling_grids_local_coord = F.affine_grid(transform_parameters, torch.Size((transform_parameters.size(0), 1, self.out_grid_size.h, self.out_grid_size.w)), align_corners=True)
        assert resampling_grids_local_coord.ndimension() == 4 and resampling_grids_local_coord.size(-1) == 2 and resampling_grids_local_coord.size(-2) == self.out_grid_size.w and resampling_grids_local_coord.size(-3) == self.out_grid_size.h, f"resampling_grids_local_coord should be of size batch_size x out_grid_width x out_grid_height x 2, but have {resampling_grids_local_coord.size()}"
    
        resampling_grids_local_coord = resampling_grids_local_coord.view(batch_size, fm_height, fm_width, self.out_grid_size.h, self.out_grid_size.w, 2)
    
        return resampling_grids_local_coord


def spatial_norm(feature_mask):
    mask_size = feature_mask.size()
    feature_mask = feature_mask.view(mask_size[0], mask_size[1], -1)
    feature_mask = feature_mask / (feature_mask.sum(dim=2, keepdim=True))
    feature_mask = feature_mask.view(mask_size)
    return feature_mask


class OTSHeadCreator(nn.Module):
    def __init__(self, aligner, feature_map_stride, feature_map_receptive_field):
        super(OTSHeadCreator, self).__init__()
        self.aligner = aligner

        rec_field, stride = self.get_rec_field_and_stride_after_concat_nets(feature_map_receptive_field, feature_map_stride,
                                                                             self.aligner.network_receptive_field, self.aligner.network_stride)
        self.box_grid_generator_image_level = BoxGridGenerator(box_size=rec_field, box_stride=stride)
        self.box_grid_generator_feature_map_level = BoxGridGenerator(box_size=self.aligner.network_receptive_field,
                                                                     box_stride=self.aligner.network_stride)

    @staticmethod
    def get_rec_field_and_stride_after_concat_nets(receptive_field_netA, stride_netA, receptive_field_netB, stride_netB):
        if isinstance(receptive_field_netA, FeatureMapSize):
            assert isinstance(stride_netA, FeatureMapSize) and isinstance(receptive_field_netB, FeatureMapSize) and isinstance(stride_netB, FeatureMapSize), "All inputs should be either of type FeatureMapSize or int"
            rec_field_w, stride_w = OTSHeadCreator.get_rec_field_and_stride_after_concat_nets(receptive_field_netA.w, stride_netA.w, receptive_field_netB.w, stride_netB.w)
            rec_field_h, stride_h = OTSHeadCreator.get_rec_field_and_stride_after_concat_nets(receptive_field_netA.h, stride_netA.h, receptive_field_netB.h, stride_netB.h)
            return FeatureMapSize(w=rec_field_w, h=rec_field_h), FeatureMapSize(w=stride_w, h=stride_h)
    
        rec_field = stride_netA * (receptive_field_netB - 1) + receptive_field_netA
        stride = stride_netA * stride_netB
        return rec_field, stride

    @staticmethod
    def resize_feature_maps_to_reference_size(ref_size, feature_maps):
        feature_maps_ref_size = []
        for fm in feature_maps:
            assert fm.size(0) == 1, f"Can process only batches of size 1, but have {fm.size(0)}"
            num_feature_channels = fm.size(1)
            identity = torch.tensor([[1, 0, 0], [0, 1, 0]], device=fm.device, dtype=fm.dtype)
            grid_size = torch.Size([1, num_feature_channels, ref_size.h, ref_size.w])
            resampling_grid = F.affine_grid(identity.unsqueeze(0), grid_size, align_corners=True)
            fm_ref_size = F.grid_sample(fm, resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            feature_maps_ref_size.append(fm_ref_size)
    
        return torch.cat(feature_maps_ref_size, dim=0)

    def create_OTS_head(self, class_feature_maps):
        reference_feature_map_size = self.aligner.reference_feature_map_size
        class_feature_maps_ref_size = self.resize_feature_maps_to_reference_size(reference_feature_map_size, class_feature_maps)
        return OTSHead(class_feature_maps_ref_size, self.aligner, self.box_grid_generator_image_level, self.box_grid_generator_feature_map_level)


class OTSHead(nn.Module):
    def __init__(self, class_feature_maps, aligner,
                       box_grid_generator_image_level,
                       box_grid_generator_feature_map_level,
                       pool_border_width=2):
        super(OTSHead, self).__init__()
        self.class_feature_maps = class_feature_maps
        self.class_batch_size = self.class_feature_maps.size(0)
        self.box_grid_generator_image_level = box_grid_generator_image_level
        self.box_grid_generator_feature_map_level = box_grid_generator_feature_map_level
        self.class_feature_maps = normalize_feature_map_L2(self.class_feature_maps, 1e-5)
        self.class_pool_mask = torch.zeros( (self.class_feature_maps.size(0), 1,
                                             self.class_feature_maps.size(2), self.class_feature_maps.size(3)), 
                                             dtype=torch.float, device=self.class_feature_maps.device)
        self.class_pool_mask[:, :,
                             pool_border_width : self.class_pool_mask.size(-2) - pool_border_width,
                             pool_border_width : self.class_pool_mask.size(-1) - pool_border_width] = 1
        self.class_pool_mask = spatial_norm(self.class_pool_mask)
        self.aligner = aligner

    def forward(self, feature_maps):
        batch_size = feature_maps.size(0)
        feature_dim = feature_maps.size(1)
        image_fm_size = FeatureMapSize(img=feature_maps)
        class_fm_size = FeatureMapSize(img=self.class_feature_maps)
        feature_dim_for_regression = class_fm_size.h * class_fm_size.w
    
        class_feature_dim = self.class_feature_maps.size(1)
        assert feature_dim == class_feature_dim, f"Feature dimensionality of input={feature_dim} and class={class_feature_dim} feature maps has to equal"
    
        feature_maps = normalize_feature_map_L2(feature_maps, 1e-5)
    
        corr_maps = torch.einsum("bfhw,afxy->abwhxy", self.class_feature_maps, feature_maps)
        corr_maps = corr_maps.contiguous().view(batch_size * self.class_batch_size, feature_dim_for_regression, image_fm_size.h, image_fm_size.w)
    
        resampling_grids_local_coord = self.aligner(corr_maps)
        cor_maps_for_recognition = corr_maps.contiguous().view(batch_size, self.class_batch_size, feature_dim_for_regression, image_fm_size.h, image_fm_size.w)
        resampling_grids_local_coord = resampling_grids_local_coord.contiguous().view(batch_size, self.class_batch_size, image_fm_size.h, image_fm_size.w, self.aligner.out_grid_size.h, self.aligner.out_grid_size.w, 2)
    
        default_boxes_xyxy_wrt_fm = self.box_grid_generator_feature_map_level.create_strided_boxes_columnfirst(fm_size=image_fm_size).view(1, 1, image_fm_size.h, image_fm_size.w, 4).to(resampling_grids_local_coord.device)
        resampling_grids_fm_coord = convert_box_coordinates_local_to_global(resampling_grids_local_coord, default_boxes_xyxy_wrt_fm)
    
        resampling_grids_fm_coord_unit = torch.cat([resampling_grids_fm_coord.narrow(-1,0,1) / (image_fm_size.w - 1) * 2 - 1, resampling_grids_fm_coord.narrow(-1,1,1) / (image_fm_size.h - 1) * 2 - 1], dim=-1).clamp(-1, 1)
    
        output_recognition = self.resample_of_correlation_map_fast(cor_maps_for_recognition, resampling_grids_fm_coord_unit, self.class_pool_mask)
        output_recognition_transform_detached = self.resample_of_correlation_map_fast(cor_maps_for_recognition, resampling_grids_fm_coord_unit.detach(), self.class_pool_mask) if output_recognition.requires_grad else output_recognition
    
        default_boxes_xyxy_wrt_image = self.box_grid_generator_image_level.create_strided_boxes_columnfirst(fm_size=image_fm_size).view(1, 1, image_fm_size.h, image_fm_size.w, 4).to(resampling_grids_local_coord.device)
        resampling_grids_image_coord = convert_box_coordinates_local_to_global(resampling_grids_local_coord, default_boxes_xyxy_wrt_image)
    
        num_pooled_points = self.aligner.out_grid_size.w * self.aligner.out_grid_size.h
        resampling_grids_x = resampling_grids_image_coord.narrow(-1, 0, 1).contiguous().view(-1, num_pooled_points)
        resampling_grids_y = resampling_grids_image_coord.narrow(-1, 1, 1).contiguous().view(-1, num_pooled_points)
        class_boxes_xyxy = torch.stack([resampling_grids_x.min(dim=1)[0],
                                        resampling_grids_y.min(dim=1)[0],
                                        resampling_grids_x.max(dim=1)[0],
                                        resampling_grids_y.max(dim=1)[0]], 1)

        corner_coordinates = resampling_grids_image_coord[:,:,:,:,[0,-1]][:,:,:,:,:,[0,-1]]
        corner_coordinates = corner_coordinates.detach_()
        corner_coordinates = corner_coordinates.view(batch_size, self.class_batch_size, image_fm_size.h, image_fm_size.w, 8) 
        corner_coordinates = corner_coordinates.transpose(3, 4).transpose(2, 3) 

        class_boxes = BoxList(class_boxes_xyxy.view(-1, 4), image_fm_size, mode="xyxy")
        default_boxes_wrt_image = BoxList(default_boxes_xyxy_wrt_image.view(-1, 4), image_fm_size, mode="xyxy")
        default_boxes_with_image_batches = cat_boxlist([default_boxes_wrt_image] * batch_size * self.class_batch_size)

        output_localization = OTSBoxCoder.build_loc_targets(class_boxes, default_boxes_with_image_batches)
        output_localization = output_localization.view(batch_size, self.class_batch_size, image_fm_size.h, image_fm_size.w, 4)
        output_localization = output_localization.transpose(3, 4).transpose(2, 3) 

        return output_localization, output_recognition, output_recognition_transform_detached, corner_coordinates


    @staticmethod
    def resample_of_correlation_map_fast(corr_maps, resampling_grids_grid_coord, class_pool_mask):
        batch_size = corr_maps.size(0)
        class_batch_size = corr_maps.size(1)
        template_fm_size = FeatureMapSize(h=resampling_grids_grid_coord.size(-3), w=resampling_grids_grid_coord.size(-2))
        image_fm_size = FeatureMapSize(img=corr_maps)
        assert template_fm_size.w * template_fm_size.h == corr_maps.size(2), 'the number of channels in the correlation map = {0} should match the size of the resampling grid = {1}'.format(corr_maps.size(2), template_fm_size)

        corr_map_merged_y_and_id_in_corr_map = corr_maps.contiguous().view(batch_size * class_batch_size,
            1, -1, image_fm_size.w)
        y_grid, x_grid = torch.meshgrid( torch.arange(template_fm_size.h), torch.arange(template_fm_size.w) )
        index_in_corr_map = y_grid + x_grid * template_fm_size.h
        resampling_grids_grid_coord_ = resampling_grids_grid_coord.clamp(-1, 1).to(dtype=torch.double)
        resampling_grids_grid_coord_x_ = resampling_grids_grid_coord_.narrow(-1,0,1)
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_.narrow(-1,1,1)
        resampling_grids_grid_coord_y_ = (resampling_grids_grid_coord_y_ + 1) / 2 * (image_fm_size.h - 1)
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_.view( [-1] + list(index_in_corr_map.size()) )
        index_in_corr_map = index_in_corr_map.unsqueeze(0)
        index_in_corr_map = index_in_corr_map.to(device=resampling_grids_grid_coord_.device,
                                                 dtype=resampling_grids_grid_coord_.dtype)
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_ + index_in_corr_map * image_fm_size.h
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_ / (image_fm_size.h * template_fm_size.h * template_fm_size.w - 1) * 2 - 1
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_.view_as(resampling_grids_grid_coord_x_)
        resampling_grids_grid_coord_merged_y_and_id_in_corr_map = torch.cat([resampling_grids_grid_coord_x_, resampling_grids_grid_coord_y_], dim=-1)

        resampling_grids_grid_coord_merged_y_and_id_in_corr_map_1d = \
            resampling_grids_grid_coord_merged_y_and_id_in_corr_map.view(batch_size * class_batch_size, -1, 1, 2)
        matches_all_channels = F.grid_sample(corr_map_merged_y_and_id_in_corr_map.to(dtype=torch.double),
                                        resampling_grids_grid_coord_merged_y_and_id_in_corr_map_1d,
                                        mode="bilinear", padding_mode='border', align_corners=True)

        matches_all_channels = matches_all_channels.view(batch_size, class_batch_size, 1,
                                                image_fm_size.h * image_fm_size.w,
                                                template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels.to(dtype=torch.float)
        mask = class_pool_mask.view(1, class_batch_size, 1, 1, template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels * mask

        matches_pooled = matches_all_channels.sum(4)
        matches_pooled = matches_pooled.view(batch_size, class_batch_size, 1, image_fm_size.h, image_fm_size.w)
        return matches_pooled

    @staticmethod
    def resample_of_correlation_map_simple(corr_maps, resampling_grids_grid_coord, class_pool_mask):
        batch_size = corr_maps.size(0)
        class_batch_size = corr_maps.size(1)
        template_fm_size = FeatureMapSize(h=resampling_grids_grid_coord.size(-3), w=resampling_grids_grid_coord.size(-2))
        image_fm_size = FeatureMapSize(img=corr_maps)
        assert template_fm_size.w * template_fm_size.h == corr_maps.size(2), f'the number of channels in the correlation map = {corr_maps.size(2)} should match the size of the resampling grid = {template_fm_size}'
    
        corr_maps = corr_maps.view(batch_size * class_batch_size, corr_maps.size(2), image_fm_size.h, image_fm_size.w)
        resampling_grids_grid_coord = resampling_grids_grid_coord.view(batch_size * class_batch_size, image_fm_size.h, image_fm_size.w, template_fm_size.h, template_fm_size.w, 2)
        
        matches_all_channels = []
        for template_x in range(template_fm_size.w):
            for template_y in range(template_fm_size.h):
                channel_id = template_x * template_fm_size.h + template_y
                channel = corr_maps[:, channel_id:channel_id + 1, :, :]
                points = resampling_grids_grid_coord[:, :, :, template_y, template_x, :]
                matches_one_channel = F.grid_sample(channel, points, mode="bilinear", padding_mode='border', align_corners=True)
                matches_all_channels.append(matches_one_channel)
    
        matches_all_channels = torch.stack(matches_all_channels, -1).view(batch_size, class_batch_size, image_fm_size.h, image_fm_size.w, template_fm_size.h * template_fm_size.w)
        
        mask = class_pool_mask.view(1, class_batch_size, 1, 1, template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels * mask
        matches_pooled = matches_all_channels.sum(4).view(batch_size, class_batch_size, 1, image_fm_size.h, image_fm_size.w)
    
        return matches_pooled


def normalize_feature_map_L2(feature_maps, epsilon=1e-6):
    return feature_maps / (feature_maps.norm(dim=1, keepdim=True) + epsilon)

class SAtt(nn.Module):
    def __init__(self, channel, reduction):
        super(SAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel//reduction, channel, 1, bias=False))
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        y = self.sigmoid(out)
        return x * y

class QAtt(nn.Module):
    def __init__(self, kernel_size=7):
        super(QAtt, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(cat)
        return x * self.sigmoid(out)


class TransformationNet(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True, normalization='batchnorm', kernel_sizes=[7,5], channels=[128,64], input_feature_dim=15*15, num_groups=16):
        super(TransformationNet, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = input_feature_dim
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=k_size//2))

            if normalization.lower() == 'batchnorm':
                nn_modules.append(nn.BatchNorm2d(ch_out))
            elif normalization.lower() == 'groupnorm':
                nn_modules.append(nn.GroupNorm(num_groups, ch_out))

            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        self.linear = nn.Conv2d(ch_out, output_dim, kernel_size=(k_size, k_size), padding=k_size//2)
        self.sup_att = SAtt(225, 16)
        self.query_att = QAtt(kernel_size=3)
        if output_dim==6:
            # assert output_dim==6, "Implemented only for affine transform"
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
            self.linear.bias.data[0] = 1
            self.linear.bias.data[4] = 1
        elif output_dim==4:
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
            self.linear.bias.data[0] = 1
            self.linear.bias.data[2] = 1

        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, corr_maps):
        corr_maps = self.sup_att(corr_maps)
        corr_maps = self.query_att(corr_maps)
        corr_maps_norm = normalize_feature_map_L2(F.relu(corr_maps))
        corr_f = self.conv(corr_maps_norm)
        transform_params = self.linear(corr_f)
        return transform_params

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
