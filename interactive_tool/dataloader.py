import os
import numpy as np
import open3d as o3d
from utils.ply import read_ply

class InteractiveDataLoader:
    """
    Interactive Dataloader handling the saving and loading of a scene. It handles groundtruth object semantics
    and can save object semantics as well as calculate the iou with the groundtruth objects.
    The overall convention when using the Dataloader:
        - the parent folder is the dataset, no naming convention
        - in the dataset, there is a folder for each scene, "scene_..."
        - in the scene folder, you can find four different kinds of data:
            - 3d point cloud or mesh is named "scan.ply"
            - groundtruth objects is named "label.ply" (should contain a 'label' attribute that indicates the instance id of each point.)
            - user clicks are saved in 'clicks' folder, segmentation masks are saved in 'masks' folder
            - if iou shall be calculated between groundtruth and user defined objects, the logits are automatically saved in "iou_record.csv"
              (You do not need to create the file, this is done automatically.)
    """
    def __init__(self, config):
        """
        Initialize the InteractiveDataLoader with the given config.

        :param config: The configuration to use.
        """
        self.config = config
        self.dataset_path = config.dataset_scenes
        self.user_point_type = config.point_type
        self.scene_names = []

        # Iterate over all directories in the dataset path
        for scene_dir in sorted(os.listdir(self.dataset_path)):
            scene_dir_path = os.path.join(self.dataset_path, scene_dir)

            # Check if the directory is a scene directory (starts with "scene_")
            dir_name_split = scene_dir.split("_")
            if os.path.isdir(scene_dir_path) and dir_name_split[0] == "scene":
                self.scene_names.append(os.path.splitext("_".join(dir_name_split[1::]))[0])

        # Initialize the current scene index
        self.__clear_curr_status()
        self.__index = 0

        # Load the first scene
        self.load_scene(0)
    
    def __clear_curr_status(self):
        """
        Resets the current scene status.

        Resets the following attributes to their initial state:
            - scene_object_names: a list of strings containing the names of the objects in the current scene
            - scene_object_semantics: a list of strings containing the semantics of the objects in the current scene
            - scene_groundtruth_object_names: a list of strings containing the names of the groundtruth objects in the current scene
            - scene_groundtruth_object_masks: a list of lists containing the masks of the groundtruth objects in the current scene
            - scene_iou: a pandas dataframe containing the iou between the groundtruth objects and the user defined objects in the current scene
            - scene_groundtruth_iou_per_object: a list of lists containing the iou between each groundtruth object and the user defined objects in the current scene
            - scene_3dpoints: a 3d point cloud or mesh representing the current scene
            - point_type: a string indicating the type of the 3d points (either 'mesh' or 'pointcloud')
        """
        self.scene_object_names = []
        self.scene_object_semantics = []
        self.scene_groundtruth_object_names = []
        self.scene_groundtruth_object_masks = [] # is later converted to np ndarray
        self.scene_iou = None # is loaded as a pd dataframe
        self.scene_groundtruth_iou_per_object = []
        self.scene_3dpoints = None
        self.point_type = None # e.g. mesh or point cloud
    
    def get_curr_scene(self):
        """returns the scene_name, scene 3d points, list of object names"""
        return self.scene_names[self.__index], self.point_type, self.scene_3dpoints, self.labels_full_ori, self.record_path, self.mask_folder, self.click_folder, [self.underscore_to_blank(name) for name in self.scene_object_names]
    
    def get_curr_scene_id(self):
        return self.__index

    def load_scene(self, idx):
        """
        Given the scene name, returns the scene name, 3d points, list of object names

        :param idx: The index of the scene to load
        :return: The scene name, the 3d points, the list of object names
        """
        # clear current lists
        name = self.scene_names[idx]
        self.__clear_curr_status()
        
        # load scene 3d points
        scene_dir = os.path.join(self.dataset_path, "scene_" + name)
        scene_3dpoints_file = os.path.join(scene_dir, 'scan.ply')

        ### set up place to save recording
        self.exp_folder = os.path.join(scene_dir, self.config.user_name)
        self.record_path = os.path.join(self.exp_folder, "iou_record.csv")
        self.mask_folder  = os.path.join(self.exp_folder, 'masks')
        self.click_folder = os.path.join(self.exp_folder, 'clicks')

        os.makedirs(self.exp_folder, exist_ok = True)
        os.makedirs(self.mask_folder, exist_ok = True)
        os.makedirs(self.click_folder, exist_ok = True)
        
        # load groundtruth labels
        if not os.path.exists(os.path.join(scene_dir, 'label.ply')):
            self.labels_full_ori = None
        else:
            point_cloud = read_ply(os.path.join(scene_dir, 'label.ply'))
            self.labels_full_ori = point_cloud['label'].astype(np.int32)
        
        # with open(scene_3dpoints_file, 'rb') as f:
        pcd_type = o3d.io.read_file_geometry_type(scene_3dpoints_file)
        if self.user_point_type is not None and self.user_point_type.lower() == "mesh" and not pcd_type == o3d.io.FileGeometry.CONTAINS_TRIANGLES:
            print("[USER WARNING] You specified the point type to be a mesh, but only a point cloud was found...using point cloud")
        elif self.user_point_type is not None and self.user_point_type.lower() == "pointcloud":
            pcd_type = o3d.io.FileGeometry.CONTAINS_POINTS
        elif self.user_point_type is not None:
            pcd_type = o3d.io.read_file_geometry_type(scene_3dpoints_file)
            print("[USER WARNING] User given preference for point type is unknown. Loading automatic type..")
        
        if pcd_type == o3d.io.FileGeometry.CONTAINS_TRIANGLES:
            self.scene_3dpoints = o3d.io.read_triangle_mesh(scene_3dpoints_file)
            self.point_type = "mesh"
        elif pcd_type == o3d.io.FileGeometry.CONTAINS_POINTS:
            self.scene_3dpoints = o3d.io.read_point_cloud(scene_3dpoints_file)
            self.point_type = "pointcloud"
        else:
            raise Exception(f"Data Format of 3d points in '3dpoints.ply' unknown for scene {name}")
        
        self.__index = self.scene_names.index(name)
        return name, self.point_type, self.scene_3dpoints, self.labels_full_ori, self.record_path, self.mask_folder, self.click_folder, [self.underscore_to_blank(name) for name in self.scene_object_names]

        
    def get_object_semantic(self, name):
        """
        Given an object name, returns the object's semantic segmentation mask.

        :param name: The name of the object
        :return: The object's semantic segmentation mask
        """
        # get the index of the object in the list of object names
        obj_idx = self.scene_object_names.index(self.blank_to_underscore(name))
        # copy the object's semantic segmentation mask
        obj_semantic = self.scene_object_semantics[obj_idx].copy()
        # return the object's semantic segmentation mask
        return obj_semantic
    
    def add_object(self, object_name):
        """
        Adds a new object to the current scene.

        This method takes an object name, converts it to an underscore format, and checks if it already exists in the current
        scene. If not, it creates a new semantic mask for the object and appends it to the list of scene objects.

        :param object_name: The name of the object to add.
        """
        # Convert the object name to underscore format
        object_name_underscore = self.blank_to_underscore(object_name)

        # Check if the object already exists in the scene
        if object_name_underscore in self.scene_object_names:
            return

        # Determine the shape of the semantic mask based on the point type
        shape = np.shape(np.asarray(self.scene_3dpoints.points if self.point_type == "pointcloud" else self.scene_3dpoints.vertices)[:, 0])

        # Create a new semantic mask with the determined shape and add it to the semantics list
        self.scene_object_semantics.append(np.zeros(shape, dtype=np.ubyte))

        # Append the new object's name to the list of object names
        self.scene_object_names.append(object_name_underscore)
    
    def update_object(self, object_name, semantic, num_new_clicks):
        """
        Given an object name and the object's mask, overwrites the existing object mask.

        :param object_name: The name of the object to update
        :param semantic: The new object mask
        :param num_new_clicks: The number of new clicks that were made to update the object mask
        :return:
        """
        # Convert the object name to underscore format
        object_name_underscore = self.blank_to_underscore(object_name)
        # Get the index of the object in the list of object names
        obj_idx = self.scene_object_names.index(object_name_underscore)
        # Verify that the new semantic mask has the same shape as the existing one
        assert(self.scene_object_semantics[obj_idx].shape == semantic.shape)
        # Overwrite the existing object mask with the new one
        self.scene_object_semantics[obj_idx] = semantic.copy()

    def get_occupied_points_idx_except_curr(self, curr_object_name):
        """
        Returns a boolean mask indicating 'occupied' points that belong 
        to at least one object, excluding the current object.

        :param curr_object_name: The name of the current object to exclude
        :return: A boolean mask where True indicates the point is occupied by other objects
        """
        # Convert the current object name to underscore format and find its index
        obj_idx = self.scene_object_names.index(self.blank_to_underscore(curr_object_name))
        
        # Create a copy of the semantic masks and remove the current object's mask
        other_objects = self.scene_object_semantics.copy()
        other_objects.pop(obj_idx)
        
        # Create a mask that is True for points occupied by any of the other objects
        mask = np.logical_or.reduce(np.ma.masked_equal(other_objects, 1).mask)
        
        return mask

    def __iter__(self):
        return self

    def __next__(self):
        if (self.__index + 1) < len(self.scene_names):
            self.__index += 1
            return self.load_scene(self.__index)
        raise StopIteration
    
    def __len__(self):
        return len(self.scene_names)

    @staticmethod
    def blank_to_underscore(name):
        return name.replace(' ', '_')
    
    @staticmethod
    def underscore_to_blank(name):
        return name.replace('_', ' ')