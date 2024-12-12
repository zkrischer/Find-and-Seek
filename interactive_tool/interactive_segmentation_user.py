from interactive_tool.gui import InteractiveSegmentationGUI
import abc
from datetime import datetime
import numpy as np
import os
from interactive_tool.utils import *
import MinkowskiEngine as ME

from models import build_model

class UserInteractiveSegmentationModel(abc.ABC):
    def __init__(self, device, config, dataloader_test):
        """
        Initializes the user interactive segmentation model.

        Args:
            device (torch.device): The device to run the model on (CPU or CUDA).
            config (Config): Configuration object containing model and dataset parameters.
            dataloader_test (iterable): Iterable test dataloader for loading test scenes.
        """
        
        self.config = config
        self.device = device
        self.dataloader_test = iter(dataloader_test)

        # Load model
        self.pretrained_weights_file = config.pretraining_weights
        self.model = build_model(config)
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode

        # Load pretrained weights if available
        if self.pretrained_weights_file:
            weights = self.pretrained_weights_file
            map_location = 'cpu' if not torch.cuda.is_available() else None
            if map_location == 'cpu':
                print('Cuda not found, using CPU')
            model_dict = torch.load(weights, map_location)
            missing_keys, unexpected_keys = self.model.load_state_dict(model_dict['model'], strict=False)
            
            # Filter out unexpected keys not relevant for loading
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if missing_keys:
                print(f'Missing Keys: {missing_keys}')
            if unexpected_keys:
                print(f'Unexpected Keys: {unexpected_keys}')
            
            self.model.eval()  # Ensure model remains in evaluation mode

        # Initialize other class parameters
        self.num_clicks = 0
        self.coords = None
        self.pred = None
        self.labels = None
        self.feats = None

        # Set the size of the segmentation mask for a single selected point
        self.cube_size = self.config.cubeedge if hasattr(config, "cubeedge") else 0.1

        # Additional attributes for visualization and mask handling
        self.visualizer = None
        self.grey_mask = None  # For points occupied by other objects
        self.object_mask = None
        self.scene_name = None
        self.object_name = None
        self.scene_point_type = None

        # Quantization size for point cloud voxelization
        self.quantization_size = config.voxel_size
        self.click_idx = {'0': []}

    
    def get_next_click(self, click_idx, click_time_idx, click_positions, num_clicks, run_model, gt_labels=None, ori_coords=None, scene_name=None, **args):
        """
        Called by GUI to forward the clicks
        
        Args:
            click_idx (dict): contains the object id as key and the list of click indices as value
            click_time_idx (dict): contains the object id as key and the list of click time indices as value
            click_positions (dict): contains the object id as key and the list of click positions as value
            num_clicks (int): the total number of clicks for all objects
            run_model (bool): whether to run the model or not
            gt_labels (torch.tensor): ground truth labels
            ori_coords (torch.tensor): the original coordinates of the scene
            scene_name (str): the name of the scene
        """
        
        if run_model:
            # if it's the first click, do nothing
            if num_clicks == 0:
                return
            else:
                self.click_idx = click_idx

                # Forward pass through the model
                outputs = self.model.forward_mask(self.pcd_features, self.aux, self.coordinates, 
                                                  self.pos_encodings_pcd, click_idx=[self.click_idx], 
                                                  click_time_idx=[click_time_idx])
                
                pred = outputs['pred_masks'][0].argmax(1)
                
                # Overrides predictions at clicks to ensure they equal the 
                # chosen object id
                for obj_id, cids in self.click_idx.items():
                    pred[cids] = int(obj_id)

                # Get the full prediction by mapping back to the original 
                # coordinates
                pred_full = pred[self.inverse_map]
                self.object_mask[:,0] = pred_full.cpu().numpy()

                # Computes the mean IoU if we have ground truth information
                if gt_labels is not None:
                    sample_iou, _ = mean_iou_scene(pred_full, gt_labels)
                    sample_iou =  str(round(sample_iou.tolist()*100,1))
                else:
                    sample_iou = 'NA'

                # Record the result to a file
                f = open(self.record_file, 'a')
                now = datetime.now()

                num_obj = len(click_idx.keys())-1
                num_click = sum([len(c) for c in click_idx.values()])
                
                line = now.strftime("%Y-%m-%d-%H-%M-%S") + '  ' + scene_name + '  NumObjects:' + str(num_obj) + '  AvgNumClicks:' + str(round(num_click/num_obj,1)) + '  mIoU:' + sample_iou + '\n'
                
                f.write(line)
                f.close()

                # Save the object mask and click positions to a file
                np.save(os.path.join(self.mask_folder, 'mask_'+ str(round(num_click/num_obj,1)) + '_' +sample_iou), 
                                     pred_full.cpu().numpy())
                np.save(os.path.join(self.click_folder, 'click_'+ str(round(num_click/num_obj,1)) + '_' + sample_iou), 
                        {
                            'click_idx': click_idx,
                            'click_time': click_time_idx
                        })

                print(line)
        
        # update gui and save new object mask
        self.visualizer.update_colors(colors=self.get_colors(reload_masks=False)) # self.object_mask is already up to date
        negative_semantic = self.object_mask[:, 2].copy() * 2 # negative semantic is 2, positive semantic is 1
        object_semantic = self.object_mask[:, 0] + negative_semantic # as these channels are mutually exclusive, we get 0 for uncertain, 1 for positive and 2 for negative

        # update the object mask in the dataloader
        self.dataloader_test.update_object(self.object_name, object_semantic, num_new_clicks=num_clicks)

    def reset_masks(self):
        """Reset the object mask and other related variables for the current object
        
        This function is called when the user selects a new object in the GUI, 
        or when the user wants to reset the object mask for some reason.
        
        It resets the object mask to zero, and also reset the grey mask to zero
        and the click indices to empty lists.
        
        """
        # mask: 0 for uncertain, 1 for positive, 2 for negative
        self.object_mask = np.zeros([np.shape(self.original_colors)[0], 3], dtype=np.uint8)
        # grey mask: 0 for uncertain, 1 for positive/negative
        self.grey_mask = np.zeros([np.shape(self.original_colors)[0]], dtype=bool)
        # click indices for the positive class
        self.pc_clicks_idx = []
        # click indices for the negative class
        self.nc_clicks_idx = []
    
    def get_colors(self, reload_masks=False):
        """
        Returns the color array for the GUI based on the object mask and semantic segmentation
        
        Args:
            reload_masks (bool): Whether to reload the masks from the dataloader or not. Defaults to False.
        
        Returns:
            colors (np.ndarray): The color array for the GUI
        """
        if reload_masks:
            # reset the object mask and grey mask
            self.reset_masks()
            # get the occupied points idx except for the current object
            self.grey_mask = self.dataloader_test.get_occupied_points_idx_except_curr(self.object_name)
            # get the semantic segmentation for the current object
            object_semantic = self.dataloader_test.get_object_semantic(self.object_name)
        
        # initialize the colors array with the original colors
        colors = self.original_colors.copy()

        # get the object ids from the object mask
        obj_ids = np.unique(self.object_mask[:, 0])
        # remove the background class (0)
        obj_ids = [obj_id for obj_id in obj_ids if obj_id != 0]
        # loop over the object ids and set the colors for each object
        for obj_id in obj_ids:
            # get the mask for the current object
            obj_mask = self.object_mask[:, 0] == obj_id
            # set the colors for the current object
            colors[obj_mask] = get_obj_color(obj_id, normalize=True)

        return colors
    
    def check_previous_next_scene(self):
        """
        Determine the availability of previous and next scenes.

        This function checks if there are scenes before or after the current scene
        in the dataloader. It is used by the GUI to enable or disable navigation buttons.

        Returns:
            tuple: A tuple containing three elements:
                - previous (bool): True if a previous scene exists, False otherwise.
                - nxt (bool): True if a next scene exists, False otherwise.
                - curr_scene_idx (int): The index of the current scene.
        """
        num_scenes = len(self.dataloader_test)  # Total number of scenes
        curr_scene_idx = self.dataloader_test.get_curr_scene_id()  # Current scene index

        # Determine if a previous scene is available
        previous = curr_scene_idx > 0

        # Determine if a next scene is available
        nxt = curr_scene_idx < num_scenes - 1

        # Return a tuple with the availability of previous and next scenes, and the current scene index
        return previous, nxt, curr_scene_idx
    
    def set_slider(self, slider_value):
        """
        Update the cube size based on the slider value.

        Args:
            slider_value (float): The new size for the segmentation cube.
        """

        self.cube_size = slider_value
    
    def run_segmentation(self):
        """
        This function sets up the segmentation environment by initializing colors, coordinates,
        and masks, and computes the backbone features for the model. It then runs the GUI
        for interactive segmentation.

        Returns:
            None
        """
        # Retrieve current scene information and paths for recording results
        self.scene_name, self.scene_point_type, self.points, labels_full_ori, record_file, mask_folder, click_folder, objects = self.dataloader_test.get_curr_scene()
        self.record_file = record_file
        self.mask_folder = mask_folder
        self.click_folder = click_folder

        # Extract and copy colors and coordinates from the scene
        # I believe this works differently depending on if it's a pointcloud or mesh
        colors_full = np.asarray(self.points.vertex_colors if self.scene_point_type == "mesh" else self.points.colors).copy()
        coords_full = np.array(self.points.points if self.scene_point_type == "pointcloud" else self.points.vertices)

        # Initialize original colors and coordinates
        self.original_colors = colors_full
        self.coords = coords_full

        # Reset masks for segmentation (Cause it's a new scene you're looking at)
        self.reset_masks()

        # Quantize coordinates for sparse representation
        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=self.coords,
            quantization_size=self.quantization_size,
            return_index=True,
            return_inverse=True)

        # Store quantized coordinates and colors
        self.coords_qv = coords_qv
        self.colors_qv = torch.from_numpy(colors_full[unique_map]).float()

        # Process original labels if available
        # I guess this is applicable if you have GT?
        if labels_full_ori is not None:
            self.labels_full_ori = torch.from_numpy(labels_full_ori).float().to(self.device)
            self.labels_qv_ori = self.labels_full_ori[unique_map]
        else:
            self.labels_full_ori = None
            self.labels_qv_ori = None

        # Store inverse map and raw quantized coordinates on the device
        self.inverse_map = inverse_map.to(self.device)
        self.raw_coords_qv = torch.from_numpy(coords_full[unique_map]).float().to(self.device)

        # Compute backbone features using the model
        # This only has to happen once due to the way the system handles clicks
        data = ME.SparseTensor(
            coordinates=ME.utils.batched_coordinates([self.coords_qv]),
            features=self.colors_qv,
            device=self.device
        )
        self.pcd_features, self.aux, self.coordinates, self.pos_encodings_pcd = self.model.forward_backbone(data, raw_coordinates=self.raw_coords_qv)

        # Initialize the visualizer for interactive segmentation
        self.visualizer = InteractiveSegmentationGUI(self)

        # Determine initial colors and object name if objects are present
        if len(objects) != 0:
            self.object_name = objects[0]
            colors = self.get_colors(reload_masks=True)
        else:
            colors = self.get_colors()

        # Run the GUI for interactive segmentation
        self.visualizer.run(scene_name=self.scene_name, point_object=self.points, coords=self.coords,
                            coords_qv=self.raw_coords_qv, colors=colors, original_colors=self.original_colors,
                            original_labels=self.labels_full_ori, original_labels_qv=self.labels_qv_ori,
                            is_point_cloud=self.scene_point_type == "pointcloud", object_names=objects)

    def load_next_scene(self, quit = False, previous=False):
        """
        Load the next scene in the dataloader (or previous scene if previous is True).
        If quit is True, the interactive segmentation will be stopped.
        """
        self.num_clicks = 0
        if quit:
            # eventually still relevant for pyviz
            return
        prev, nxt, curr_scene_idx = self.check_previous_next_scene()
        if not previous and nxt:
            # load next scene
            self.scene_name, self.scene_point_type, self.points, labels_full_ori, record_file, mask_folder, click_folder, objects = next(self.dataloader_test)
            self.reset_masks()
        elif previous and prev:
            # load previous scene
            self.scene_name, self.scene_point_type, self.points, labels_full_ori, record_file, mask_folder, click_folder, objects = self.dataloader_test.load_scene(curr_scene_idx-1)
            self.reset_masks()
        else: 
            return
        
        self.record_file = record_file
        self.mask_folder  = mask_folder
        self.click_folder = click_folder
        
        # Get the colors and coordinates of the point cloud
        colors_full = np.asarray(self.points.vertex_colors if self.scene_point_type == "mesh" else self.points.colors).copy()
        coords_full = np.array(self.points.points if self.scene_point_type == "pointcloud" else self.points.vertices)
        
        # Store the original colors
        self.original_colors = colors_full
        
        # Store the coordinates
        self.coords = coords_full

        # Reset the masks
        self.reset_masks()

        ### quantization
        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
                coordinates=self.coords,
                quantization_size=self.quantization_size,
                return_index=True,
                return_inverse=True)
        
        # Store the quantized coordinates and their inverse map
        self.coords_qv = coords_qv
        self.colors_qv = torch.from_numpy(colors_full[unique_map]).float()
        self.inverse_map = inverse_map.to(self.device)
        self.raw_coords_qv = torch.from_numpy(coords_full[unique_map]).float().to(self.device)
        
        ### compute backbone features
        data = ME.SparseTensor(
                            coordinates=ME.utils.batched_coordinates([self.coords_qv]),
                            features=self.colors_qv,
                            device=self.device
                            )
        self.pcd_features, self.aux, self.coordinates, self.pos_encodings_pcd = self.model.forward_backbone(data, raw_coordinates=self.raw_coords_qv)

        if len(objects) != 0: # init current object to the first one if there are already objects in the data set
            self.object_name = objects[0]
            colors = self.get_colors(reload_masks=True)
        else:
            self.object_name = None
            self.reset_masks()
            colors = self.get_colors()

        # Basically has prepped the next scene the same way as it did the 
        # initial scene, and then sets it up for the visualizer
        self.visualizer.set_new_scene(scene_name = self.scene_name, point_object = self.points, coords = self.coords, 
                                      coords_qv = self.raw_coords_qv, colors = colors, original_colors = self.original_colors, 
                                      original_labels = self.labels_full_ori, original_labels_qv = self.labels_qv_ori, 
                                      is_point_cloud=self.scene_point_type=="pointcloud", object_names=objects)
        return
    
    def load_object(self, object_name, load_colors=True):
        """
        Is called by GUI to either load an existing object or to create a new object
        
        Args:
            object_name (str): The name of the object to be loaded
            load_colors (bool): Whether to load the colors of the object. Defaults to True.
        """
        # Load the object
        self.object_name = object_name
        self.dataloader_test.add_object(object_name) # does nothing if object already exists
        
        # Get the colors of the object
        colors = self.get_colors(reload_masks=True) if load_colors else None
        
        # Select the object in the visualizer
        self.visualizer.select_object(colors=colors)
