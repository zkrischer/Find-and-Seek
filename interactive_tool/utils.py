try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')
import torch

# constants and flags
USE_TRAINING_CLICKS = False
OBJECT_CLICK_COLOR = [0.2, 0.81, 0.2] # colors between 0 and 1 for open3d
BACKGROUND_CLICK_COLOR = [0.81, 0.2, 0.2] # colors between 0 and 1 for open3d
UNSELECTED_OBJECTS_COLOR = [0.4, 0.4, 0.4]
SELECTED_OBJECT_COLOR = [0.2, 0.81, 0.2]
obj_color = {1: [1, 211, 211], 2: [233,138,0], 3: [41,207,2], 4: [244, 0, 128], 5: [194, 193, 3], 6: [121, 59, 50],
             7: [254, 180, 214], 8: [239, 1, 51], 9: [125, 0, 237], 10: [229, 14, 241]}

def get_obj_color(obj_idx, normalize=False):
    """
    Get the color for a given object index

    Args:
        obj_idx (int): The object index
        normalize (bool, optional): Whether to normalize the color to the range [0, 1]. Defaults to False.

    Returns:
        list: The color as a list of three floats
    """
    # get the color from the obj_color dictionary
    r, g, b = obj_color[obj_idx]

    # if normalize is True, normalize the color to the range [0, 1]
    if normalize:
        r /= 256
        g /= 256
        b /= 256

    # return the color as a list of three floats
    return [r, g, b]

def find_nearest(coordinates, value):
    """
    Find the index of the nearest coordinate to the given value

    Args:
        coordinates (torch.Tensor): The coordinates to search in
        value (list or torch.Tensor): The value to search for

    Returns:
        int: The index of the nearest coordinate
    """
    # calculate the distance between the coordinates and the given value
    distance = torch.cdist(coordinates, torch.tensor([value]).to(coordinates.device), p=2)
    # find the index of the nearest coordinate
    return distance.argmin().tolist()

def mean_iou_single(pred, labels):
    """
    Calculate the mean IoU for a single object

    Args:
        pred (torch.Tensor): The predicted mask
        labels (torch.Tensor): The ground truth mask

    Returns:
        float: The mean IoU
    """
    # calculate the true positive pixels
    truepositive = pred*labels
    # calculate the intersection between the predicted and ground truth masks
    intersection = torch.sum(truepositive==1)
    # calculate the union between the predicted and ground truth masks
    uni = torch.sum(pred==1) + torch.sum(labels==1) - intersection
    # calculate the mean IoU
    iou = intersection/uni
    return iou

def mean_iou_scene(pred, labels):
    """
    Calculate the mean IoU for a single scene

    Args:
        pred (torch.Tensor): The predicted mask
        labels (torch.Tensor): The ground truth mask

    Returns:
        tuple: A tuple containing the mean IoU as a float, and a dictionary with the IoU of each object
    """
    # calculate the object IDs (excluding the background)
    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids!=0]
    # calculate the number of objects
    obj_num = len(obj_ids)
    # calculate the sum of the IoU of all objects
    iou_sample = 0.0
    # create a dictionary to store the IoU of each object
    iou_dict = {}
    # loop through each object ID
    for obj_id in obj_ids:
        # calculate the IoU of the current object
        obj_iou = mean_iou_single(pred==obj_id, labels==obj_id)
        # store the IoU in the dictionary
        iou_dict[int(obj_id)] = float(obj_iou)
        # add the IoU to the sum
        iou_sample += obj_iou

    # calculate the mean IoU
    iou_sample /= obj_num

    # return the mean IoU and the dictionary of IoU of each object
    return iou_sample, iou_dict
