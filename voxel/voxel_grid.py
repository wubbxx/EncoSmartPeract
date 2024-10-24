# Voxelizer modified from ARM for DDP training
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

from functools import reduce
from operator import mul

import torch
from torch import nn

MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False


class VoxelGrid(nn.Module):

    def __init__(self,
                 coord_bounds,
                 voxel_size: int,
                 device,
                 batch_size,
                 feature_size,  # e.g. rgb or image features
                 max_num_coords: int,):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = [voxel_size] * 3
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = torch.tensor(self._voxel_shape,
                                              ).unsqueeze(
            0) + 2  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(coord_bounds, dtype=torch.float,
                                          ).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [torch.tensor([batch_size], ), max_dims,
             torch.tensor([4 + feature_size], )], -1).tolist()

        self.register_buffer('_ones_max_coords', torch.ones((batch_size, max_num_coords, 1)))
        self._num_coords = max_num_coords

        shape = self._total_dims_list
        result_dim_sizes = torch.tensor(
            [reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [1], )
        self.register_buffer('_result_dim_sizes', result_dim_sizes)
        flat_result_size = reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float)
        flat_output = torch.ones(flat_result_size, dtype=torch.float) * self._initial_val
        self.register_buffer('_flat_output', flat_output)

        self.register_buffer('_arange_to_max_coords', torch.arange(4 + feature_size))
        self._flat_zeros = torch.zeros(flat_result_size, dtype=torch.float)

        self._const_1 = torch.tensor(1.0, )
        self._batch_size = batch_size

        # Coordinate Bounds:
        bb_mins = self._coord_bounds[..., 0:3]
        self.register_buffer('_bb_mins', bb_mins)
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - bb_mins
        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = self._voxel_shape_spec.int()
        dims_orig = self._voxel_shape_spec.int() - 2
        self.register_buffer('_dims_orig', dims_orig)

        # self._dims_m_one = (dims - 1).int()
        dims_m_one = (dims - 1).int()
        self.register_buffer('_dims_m_one', dims_m_one)

        # BS x 1 x 3
        res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)
        self.register_buffer('_res', res)

        voxel_indicy_denmominator = res + MIN_DENOMINATOR
        self.register_buffer('_voxel_indicy_denmominator', voxel_indicy_denmominator)

        self.register_buffer('_dims_m_one_zeros', torch.zeros_like(dims_m_one))

        batch_indices = torch.arange(self._batch_size, dtype=torch.int).view(self._batch_size, 1, 1)
        self.register_buffer('_tiled_batch_indices', batch_indices.repeat([1, self._num_coords, 1]))

        w = self._voxel_shape[0] + 2
        arange = torch.arange(0, w, dtype=torch.float, )
        index_grid = torch.cat([
            arange.view(w, 1, 1, 1).repeat([1, w, w, 1]),
            arange.view(1, w, 1, 1).repeat([w, 1, w, 1]),
            arange.view(1, 1, w, 1).repeat([w, w, 1, 1])], dim=-1).unsqueeze(
            0).repeat([self._batch_size, 1, 1, 1, 1])
        self.register_buffer('_index_grid', index_grid)

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(self, src: torch.Tensor, index: torch.Tensor, out: torch.Tensor,
                      dim: int = -1):
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims])
        indices_for_flat_tiled = ((indices * indices_scales).sum(
            dim=-1, keepdims=True)).view(-1, 1).repeat(
            *[1, self._voxel_feature_size])

        implicit_indices = self._arange_to_max_coords[
                           :self._voxel_feature_size].unsqueeze(0).repeat(
            *[indices_for_flat_tiled.shape[0], 1])
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates, flat_indices_for_flat,
            out=torch.zeros_like(self._flat_output))
        return flat_scatter.view(self._total_dims_list)

    def coords_to_bounding_voxel_grid(self, coords, coord_features=None, coord_bounds=None):
        # 打印初始输入数据的形状
        print(f"Initial coords shape: {coords.shape}")
        if coord_features is not None:
            print(f"Initial coord_features shape: {coord_features.shape}")
        if coord_bounds is not None:
            print(f"Initial coord_bounds shape: {coord_bounds.shape}")

        # 打印初始体素化的关键参数
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins
        print(f"Initial voxel_indicy_denmominator: {voxel_indicy_denmominator}")
        print(f"Initial resolution (res): {res}")
        print(f"Initial bounding box minimums (bb_mins): {bb_mins}")

        # 如果有边界框，调整体素的分辨率和索引计算分母
        if coord_bounds is not None:
            bb_mins = coord_bounds[..., 0:3]
            bb_maxs = coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR)
            voxel_indicy_denmominator = res + MIN_DENOMINATOR
            print(f"Adjusted bounding box minimums (bb_mins): {bb_mins}")
            print(f"Bounding box ranges (bb_ranges): {bb_ranges}")
            print(f"Adjusted resolution (res): {res}")
            print(f"Adjusted voxel_indicy_denmominator: {voxel_indicy_denmominator}")

        # 计算平移后的包围盒最小值
        bb_mins_shifted = bb_mins - res  # shift back by one
        print(f"Shifted bounding box minimums (bb_mins_shifted): {bb_mins_shifted}")

        # 计算每个点的体素索引
        floor = torch.floor((coords - bb_mins_shifted.unsqueeze(1)) / voxel_indicy_denmominator.unsqueeze(1)).int()
        print(f"Calculated floor (voxel indices before clipping) shape: {floor.shape}, data: {floor}")

        # 将体素索引裁剪到有效范围
        voxel_indices = torch.min(floor, self._dims_m_one)
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros)
        print(f"Clipped voxel indices shape: {voxel_indices.shape}, data: {voxel_indices}")

        # 初始化体素的值为原始坐标
        voxel_values = coords
        print(f"Voxel values (initial coords) shape: {voxel_values.shape}")

        # 如果有附加特征，将其添加到体素值中
        if coord_features is not None:
            voxel_values = torch.cat([voxel_values, coord_features], -1)
            print(f"Voxel values with features shape: {voxel_values.shape}")

        # 获取体素的总数量并准备将体素索引和批处理索引组合
        _, num_coords, _ = voxel_indices.shape
        print(f"Number of coordinates (num_coords): {num_coords}")

        all_indices = torch.cat([self._tiled_batch_indices[:, :num_coords], voxel_indices], -1)
        print(f"All indices shape (with batch indices): {all_indices.shape}, data: {all_indices}")

        # 将坐标和其他信息拼接成体素值
        voxel_values_pruned_flat = torch.cat([voxel_values, self._ones_max_coords[:, :num_coords]], -1)
        print(f"Voxel values pruned flat shape: {voxel_values_pruned_flat.shape}")

        # 使用 scatter_nd 操作将体素值插入到 3D 网格中
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),  # 扁平化索引
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size)  # 扁平化体素值
        )
        print(f"Scattered voxel grid shape (before trimming): {scattered.shape}")

        # 修剪体素网格的边缘
        vox = scattered[:, 1:-1, 1:-1, 1:-1]
        print(f"Trimmed voxel grid shape: {vox.shape}")

        # 如果启用了 INCLUDE_PER_VOXEL_COORD，则将每个体素的中心坐标添加为特征
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (res_centre + bb_mins_shifted.unsqueeze(1).unsqueeze(1).unsqueeze(1))[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat([vox[..., :-1], coord_positions, vox[..., -1:]], -1)
            print(f"Voxel grid with coord positions shape: {vox.shape}")

        # 计算每个体素是否被占用
        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([vox[..., :-1], occupied], -1)
        print(f"Final voxel grid with occupancy shape: {vox.shape}")

        # 添加体素索引网格信息并返回最终的体素网格
        final_vox = torch.cat([vox[..., :-1], self._index_grid[:, :-2, :-2, :-2] / self._voxel_d, vox[..., -1:]], -1)
        print(f"Final voxel grid shape with index grid: {final_vox.shape}")
        
        # 返回最终生成的体素网格
        return final_vox

        # print(f"Final voxel grid shape with index grid: {final_vox.shape}")
        # 经过体素化后，生成的 3D 体素网格形状为 [1, 100, 100, 100, 10]。
        # 这个 3D 网格中，每个体素有 10 个特征，包含 XYZ 坐标、RGB 颜色信息、
        # 体素的中心坐标以及该体素是否被占用的信息。
        # return final_vox
