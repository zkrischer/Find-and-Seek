# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.modules.attention_block import *
from models.position_embedding import PositionEmbeddingCoordsSine, PositionalEncoding3D, PositionalEncoding1D
from torch.cuda.amp import autocast
from .backbone import build_backbone

import itertools

class Agile3d(nn.Module):
    def __init__(self, backbone, hidden_dim, num_heads, dim_feedforward,
                 shared_decoder, num_decoders, num_bg_queries, dropout, pre_norm,
                 positional_encoding_type, normalize_pos_enc, hlevels,
                 voxel_size, gauss_scale, aux
                 ):
        super().__init__()

        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.hlevels = hlevels
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_bg_queries = num_bg_queries
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.pos_enc_type = positional_encoding_type
        self.aux = aux

        self.backbone = backbone

        self.lin_squeeze_head = conv(
            self.backbone.PLANES[7], self.mask_dim, kernel_size=1, stride=1, bias=True, D=3
        )

        self.bg_query_feat = nn.Embedding(num_bg_queries, hidden_dim)
        self.bg_query_pos = nn.Embedding(num_bg_queries, hidden_dim)


        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=self.mask_dim,
                                                       gauss_scale=self.gauss_scale,
                                                       normalize=self.normalize_pos_enc)
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine",
                                                       d_pos=self.mask_dim,
                                                       normalize=self.normalize_pos_enc)
        else:
            assert False, 'pos enc type not known'

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.masked_transformer_decoder = nn.ModuleList()

        # Click-to-scene attention
        self.c2s_attention = nn.ModuleList()

        # Click-to-click attention
        self.c2c_attention = nn.ModuleList()

        # FFN
        self.ffn_attention = nn.ModuleList()

        # Scene-to-click attention
        self.s2c_attention = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_c2s_attention = nn.ModuleList()
            tmp_s2c_attention = nn.ModuleList()
            tmp_c2c_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()

            for i, hlevel in enumerate(self.hlevels):
                tmp_c2s_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_s2c_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_c2c_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

            self.c2s_attention.append(tmp_c2s_attention)
            self.s2c_attention.append(tmp_s2c_attention)
            self.c2c_attention.append(tmp_c2c_attention)
            self.ffn_attention.append(tmp_ffn_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.time_encode = PositionalEncoding1D(hidden_dim, 200)


    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            ### this is a trick to bypass a bug in Minkowski Engine cpu version
            if coords[i].F.is_cuda:
                coords_batches = coords[i].decomposed_features
            else:
                coords_batches = [coords[i].F]
            for coords_batch in coords_batches:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward_backbone(self, x, raw_coordinates=None):
        """
        Runs the backbone of the network to extract features from the input point cloud

        Args:
            x: Input point cloud
            raw_coordinates: Input coordinates

        Returns:
            pcd_features: Extracted features
            aux: Auxiliary information
            coordinates: Coordinates of the input point cloud
            pos_encodings_pcd: Positional encodings of the input point cloud
        """
        pcd_features, aux = self.backbone(x)

        with torch.no_grad():
            # convert coordinates to sparse tensor
            coordinates = me.SparseTensor(features=raw_coordinates,
                                          coordinate_manager=aux[-1].coordinate_manager,
                                          coordinate_map_key=aux[-1].coordinate_map_key,
                                          device=aux[-1].device)
            # get coordinates of the input point cloud at each level
            coords = [coordinates]
            for _ in reversed(range(len(aux)-1)):
                coords.append(self.pooling(coords[-1]))

            # reverse the order of the coordinates
            coords.reverse()

        # get positional encodings of the input point cloud
        pos_encodings_pcd = self.get_pos_encs(coords)

        # squeeze the features to the correct shape
        pcd_features = self.lin_squeeze_head(pcd_features)

        return pcd_features, aux, coordinates, pos_encodings_pcd

    def forward_mask(self, pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=None, click_time_idx=None):
        """
        Forward pass of the mask module.

        Args:
            pcd_features: Features of the input point cloud
            aux: Auxiliary information
            coordinates: Coordinates of the input point cloud
            pos_encodings_pcd: Positional encodings of the input point cloud
            click_idx: Indices of the clicked points
            click_time_idx: Indices of the clicked points in time

        Returns:
            predictions_mask: Predicted masks
        """
        batch_size = pcd_features.C[:,0].max() + 1

        predictions_mask = [[] for i in range(batch_size)]
        attention_mask = [[] for i in range(batch_size)]

        bg_learn_queries = self.bg_query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        bg_learn_query_pos = self.bg_query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        for b in range(batch_size):
            # Get the minimum and maximum coordinates of the current batch
            if coordinates.F.is_cuda:
                mins = coordinates.decomposed_features[b].min(dim=0)[0].unsqueeze(0)
                maxs = coordinates.decomposed_features[b].max(dim=0)[0].unsqueeze(0)
            else:
                mins = coordinates.F.min(dim=0)[0].unsqueeze(0)
                maxs = coordinates.F.max(dim=0)[0].unsqueeze(0)

            # Get the indices of the clicked points and the time indices
            click_idx_sample = click_idx[b]
            click_time_idx_sample = click_time_idx[b]

            # Get the indices of the background points
            bg_click_idx = click_idx_sample['0']

            # Get the number of foreground objects
            fg_obj_num = len(click_idx_sample.keys()) - 1

            # Get the number of foreground queries
            fg_query_num_split = [len(click_idx_sample[str(i)]) for i in range(1, fg_obj_num+1)]
            fg_query_num = sum(fg_query_num_split)

            # Get the coordinates of the foreground points
            if coordinates.F.is_cuda:
                fg_clicks_coords = torch.vstack([coordinates.decomposed_features[b][click_idx_sample[str(i)], :]
                                                for i in range(1,fg_obj_num+1)]).unsqueeze(0)
            else:
                fg_clicks_coords = torch.vstack([coordinates.F[click_idx_sample[str(i)], :]
                                                for i in range(1,fg_obj_num+1)]).unsqueeze(0)

            # Get the positional encodings of the foreground points
            fg_query_pos = self.pos_enc(fg_clicks_coords.float(),
                                     input_range=[mins, maxs]
                                     )

            # Get the time indices of the foreground points
            fg_clicks_time_idx = list(itertools.chain.from_iterable([click_time_idx_sample[str(i)] for i in range(1,fg_obj_num+1)]))
            fg_query_time = self.time_encode[fg_clicks_time_idx].T.unsqueeze(0).to(fg_query_pos.device)
            fg_query_pos = fg_query_pos + fg_query_time

            # If there are background points, get their coordinates and positional encodings
            if len(bg_click_idx)!=0:
                if coordinates.F.is_cuda:
                    bg_click_coords = coordinates.decomposed_features[b][bg_click_idx].unsqueeze(0)
                else:
                    bg_click_coords = coordinates.F[bg_click_idx].unsqueeze(0)
                bg_query_pos = self.pos_enc(bg_click_coords.float(),
                                        input_range=[mins, maxs]
                                        )  # [num_queries, 128]
                bg_query_time = self.time_encode[click_time_idx_sample['0']].T.unsqueeze(0).to(bg_query_pos.device)
                bg_query_pos = bg_query_pos + bg_query_time

                # Concatenate the background queries with the learned background queries
                bg_query_pos = torch.cat([bg_learn_query_pos[b].T.unsqueeze(0), bg_query_pos],dim=-1)
            else:
                # If there are no background points, use the learned background queries
                bg_query_pos = bg_learn_query_pos[b].T.unsqueeze(0)

            # Permute the foreground and background queries
            fg_query_pos = fg_query_pos.permute((2, 0, 1))[:,0,:] # [num_queries, 128]
            bg_query_pos = bg_query_pos.permute((2, 0, 1))[:,0,:] # [num_queries, 128]

            # Get the number of background queries
            bg_query_num = bg_query_pos.shape[0]

            # Get the features of the foreground and background points
            if pcd_features.F.is_cuda:
                fg_queries = torch.vstack([pcd_features.decomposed_features[b][click_idx_sample[str(i)], :]
                                           for i in range(1,fg_obj_num+1)])
            else:
                fg_queries = torch.vstack([pcd_features.F[click_idx_sample[str(i)], :]
                                           for i in range(1,fg_obj_num+1)])

            if len(bg_click_idx)!=0:
                # If there are background points, get their features
                if pcd_features.F.is_cuda:
                    bg_queries = pcd_features.decomposed_features[b][bg_click_idx,:]
                else:
                    bg_queries = pcd_features.F[bg_click_idx,:]
                # Concatenate the background features with the learned background features
                bg_queries = torch.cat([bg_learn_queries[b], bg_queries], dim=0)
            else:
                # If there are no background points, use the learned background features
                bg_queries = bg_learn_queries[b]

            # Get the features of the input point cloud
            if pcd_features.F.is_cuda:
                src_pcd = pcd_features.decomposed_features[b]
            else:
                src_pcd = pcd_features.F

            refine_time = 0

            for decoder_counter in range(self.num_decoders):
                if self.shared_decoder:
                    decoder_counter = 0
                for i, hlevel in enumerate(self.hlevels):
                    # Get the positional encodings of the input point cloud
                    pos_enc = pos_encodings_pcd[hlevel][0][b]# [num_points, 128]

                    # If this is the first refinement step, set the attention mask to None
                    if refine_time == 0:
                        attn_mask = None

                    # Run the click-to-scene attention
                    output = self.c2s_attention[decoder_counter][i](
                        torch.cat([fg_queries, bg_queries],dim=0), # [num_queries, 128]
                        src_pcd, # [num_points, 128]
                        memory_mask=attn_mask,
                        memory_key_padding_mask=None,
                        pos=pos_enc, # [num_points, 128]
                        query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0) # [num_queries, 128]
                    ) # [num_queries, 128]

                    # Run the click-to-click attention
                    output = self.c2c_attention[decoder_counter][i](
                        output, # [num_queries, 128]
                        tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0) # [num_queries, 128]
                    ) # [num_queries, 128]

                    # Run the FFN
                    queries = self.ffn_attention[decoder_counter][i](
                        output
                    ) # [num_queries, 128]

                    # Run the scene-to-click attention
                    src_pcd = self.s2c_attention[decoder_counter][i](
                        src_pcd,
                        queries, # [num_queries, 128]
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=torch.cat([fg_query_pos, bg_query_pos], dim=0), # [num_queries, 128]
                        query_pos=pos_enc # [num_points, 128]
                    ) # [num_points, 128]

                    # Split the output into foreground and background queries
                    fg_queries, bg_queries = queries.split([fg_query_num, bg_query_num], 0)

                    # Run the mask module
                    outputs_mask, attn_mask = self.mask_module(
                                                        fg_queries,
                                                        bg_queries,
                                                        src_pcd,
                                                        ret_attn_mask=True,
                                                        fg_query_num_split=fg_query_num_split)

                    # Append the predicted mask to the list of predictions
                    predictions_mask[b].append(outputs_mask)
                    attention_mask[b].append(attn_mask)

                    # Increment the refinement time
                    refine_time += 1

        # Transpose the list of predictions
        predictions_mask = [list(i) for i in zip(*predictions_mask)]

        # Not sure if I want to transpose these things as well, but I'm going to in this case
        attention_mask = [list(i) for i in zip(*attention_mask)]
        # Return the predicted masks
        return predictions_mask


    def mask_module(self, fg_query_feat, bg_query_feat, mask_features, ret_attn_mask=True,
                                fg_query_num_split=None):
        """
        This function takes in the foreground and background query features, the mask features, and returns the predicted
        masks and attention masks.

        Args:
        - fg_query_feat: The foreground query features of shape [num_fg_queries, 128]
        - bg_query_feat: The background query features of shape [num_bg_queries, 128]
        - mask_features: The mask features of shape [num_points, 128]
        - ret_attn_mask: Whether to return the attention mask. Default is True.
        - fg_query_num_split: The number of foreground queries for each object. Default is None.

        Returns:
        - output_masks: The predicted mask of shape [num_points, num_objects+1]
        - attn_mask: The attention mask of shape [num_queries, num_points]
        """

        # Normalize the foreground query features
        fg_query_feat = self.decoder_norm(fg_query_feat)
        # Project the foreground query features to the mask embedding space
        fg_mask_embed = self.mask_embed_head(fg_query_feat)

        # Compute the dot product between the mask features and the foreground mask embeddings
        fg_prods = mask_features @ fg_mask_embed.T

        # Split the foreground masks into separate objects
        fg_prods = fg_prods.split(fg_query_num_split, dim=1)

        # Compute the maximum of the foreground masks for each object
        fg_masks = []
        for fg_prod in fg_prods:
            fg_masks.append(fg_prod.max(dim=-1, keepdim=True)[0])

        # Concatenate the foreground masks
        fg_masks = torch.cat(fg_masks, dim=-1)

        # Normalize the background query features
        bg_query_feat = self.decoder_norm(bg_query_feat)
        # Project the background query features to the mask embedding space
        bg_mask_embed = self.mask_embed_head(bg_query_feat)
        # Compute the dot product between the mask features and the background mask embeddings
        bg_masks = (mask_features @ bg_mask_embed.T).max(dim=-1, keepdim=True)[0]

        # Concatenate the background masks with the foreground masks
        output_masks = torch.cat([bg_masks, fg_masks], dim=-1)

        if ret_attn_mask:
            # Compute the labels for the predicted masks
            output_labels = output_masks.argmax(1)

            # Compute the attention mask for the background queries
            bg_attn_mask = ~(output_labels == 0)
            bg_attn_mask = bg_attn_mask.unsqueeze(0).repeat(bg_query_feat.shape[0], 1)
            bg_attn_mask[torch.where(bg_attn_mask.sum(-1) == bg_attn_mask.shape[-1])] = False

            # Compute the attention mask for the foreground queries
            fg_attn_mask = []
            for fg_obj_id in range(1, fg_masks.shape[-1]+1):
                fg_obj_mask = ~(output_labels == fg_obj_id)
                fg_obj_mask = fg_obj_mask.unsqueeze(0).repeat(fg_query_num_split[fg_obj_id-1], 1)
                fg_obj_mask[torch.where(fg_obj_mask.sum(-1) == fg_obj_mask.shape[-1])] = False
                fg_attn_mask.append(fg_obj_mask)

            # Concatenate the attention masks for the foreground and background queries
            fg_attn_mask = torch.cat(fg_attn_mask, dim=0)
            attn_mask = torch.cat([fg_attn_mask, bg_attn_mask], dim=0)

            return output_masks, attn_mask

        return output_masks



    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):

        return [
            {"pred_masks": a} for a in outputs_seg_masks[:-1]
        ]





def build_agile3d(args):

    backbone = build_backbone(args)

    model = Agile3d(
                    backbone=backbone, 
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads, 
                    dim_feedforward=args.dim_feedforward,
                    shared_decoder=args.shared_decoder,
                    num_decoders=args.num_decoders, 
                    num_bg_queries=args.num_bg_queries,
                    dropout=args.dropout, 
                    pre_norm=args.pre_norm, 
                    positional_encoding_type=args.positional_encoding_type,
                    normalize_pos_enc=args.normalize_pos_enc,
                    hlevels=args.hlevels, 
                    voxel_size=args.voxel_size,
                    gauss_scale=args.gauss_scale,
                    aux=args.aux
                    )

    return model
