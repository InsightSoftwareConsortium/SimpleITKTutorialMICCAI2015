import numpy as np
from numpy import linalg

import matplotlib.pyplot as plt
from IPython.display import clear_output
from ipywidgets import interact, fixed

import SimpleITK as sitk

def load_RIRE_ground_truth(file_name):
    """
    Load the point sets defining the ground truth transformations for the RIRE 
    training dataset.

    Args: 
        file_name (str): RIRE ground truth file name. File format is specific to 
                         the RIRE training data, with the actual data expectd to 
                         be in lines 15-23.
    Returns:
    Two lists of tuples representing the points in the "left" and "right" 
    coordinate systems.
    """
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
        l = []
        r = []
        # Fiducial information is in lines 15-22, starting with the second entry.
        for line in lines[15:23]:
            coordinates = line.split()
            l.append((float(coordinates[1]), float(coordinates[2]), float(coordinates[3])))
            r.append((float(coordinates[4]), float(coordinates[5]), float(coordinates[6])))
    return (l, r)


def absolute_orientation_m(points_in_left, points_in_right):
    """
    Absolute orientation using a matrix to represent the rotation. Solution is 
    due to S. Umeyama, "Least-Squares Estimation of Transformation Parameters 
    Between Two Point Patterns", IEEE Trans. Pattern Anal. Machine Intell., 
    vol. 13(4): 376-380.
    
    This is a refinement of the method proposed by Arun, Huang and Blostein, 
    ensuring that the rotation matrix is indeed a rotation and not a reflection. 
    
    Args:
        points_in_left (list(tuple)): Set of points corresponding to 
                                      points_in_right in a different coordinate system.
        points_in_right (list(tuple)): Set of points corresponding to 
                                       points_in_left in a different coordinate system.
        
    Returns:
        R,t (numpy.ndarray, numpy.array): Rigid transformation that maps points_in_left 
                                          onto points_in_right.
                                          R*points_in_left + t = points_in_right
    """
    num_points = len(points_in_left)
    dim_points = len(points_in_left[0])
    # Cursory check that the number of points is sufficient.
    if num_points<dim_points:      
        raise ValueError('Number of points must be greater/equal {0}.'.format(dim_points))

    # Construct matrices out of the two point sets for easy manipulation.
    left_mat = np.array(points_in_left).T
    right_mat = np.array(points_in_right).T
     
    # Center both data sets on the mean.
    left_mean = left_mat.mean(1)
    right_mean = right_mat.mean(1)
    left_M = left_mat - np.tile(left_mean, (num_points, 1)).T     
    right_M = right_mat - np.tile(right_mean, (num_points, 1)).T     
    
    M = left_M.dot(right_M.T)               
    U,S,Vt = linalg.svd(M)
    V=Vt.T
    # V * diag(1,1,det(U*V)) * U' - diagonal matrix ensures that we have a 
    # rotation and not a reflection.
    R = V.dot(np.diag((1,1,linalg.det(U.dot(V))))).dot(U.T) 
    t = right_mean - R.dot(left_mean) 
    return R,t


def generate_random_pointset(image, num_points):
    """
    Generate a random set (uniform sample) of points in the given image's domain.
    
    Args:
        image (SimpleITK.Image): Domain in which points are created.
        num_points (int): Number of points to generate.
        
    Returns:
        A list of points (tuples).
    """
    # Continous random uniform point indexes inside the image bounds.
    point_indexes = np.multiply(np.tile(image.GetSize(), (num_points, 1)), 
                                np.random.random((num_points, image.GetDimension())))
    pointset_list = point_indexes.tolist()
    # Get the list of physical points corresponding to the indexes.
    return [image.TransformContinuousIndexToPhysicalPoint(point_index) \
            for point_index in pointset_list]


def registration_errors(tx, reference_fixed_point_list, reference_moving_point_list, 
                        display_errors = False, figure_size=(8,6)):
  """
  Distances between points transformed by the given transformation and their
  location in another coordinate system. When the points are only used to 
  evaluate registration accuracy (not used in the registration) this is the 
  Target Registration Error (TRE).
  
  Args:
      tx (SimpleITK.Transform): The transform we want to evaluate.
      reference_fixed_point_list (list(tuple-like)): Points in fixed image 
                                                     cooredinate system.
      reference_moving_point_list (list(tuple-like)): Points in moving image 
                                                      cooredinate system.
      display_points (boolean): Display a 3D figure with lines connecting 
                                corresponding points.

  Returns:
   (mean, std, min, max, errors) (float, float, float, float, [float]): 
    TRE statistics and original TREs.
  """
  transformed_fixed_point_list = [tx.TransformPoint(p) for p in reference_fixed_point_list]

  errors = [linalg.norm(np.array(p_fixed) -  np.array(p_moving))
            for p_fixed,p_moving in zip(transformed_fixed_point_list, reference_moving_point_list)]
  min_errors = np.min(errors)
  max_errors = np.max(errors)
  if display_errors:
      from mpl_toolkits.mplot3d import Axes3D
      import matplotlib.pyplot as plt
      import matplotlib
      fig = plt.figure(figsize=figure_size)
      ax = fig.add_subplot(111, projection='3d')
   
      collection = ax.scatter(list(np.array(reference_fixed_point_list).T)[0],
                              list(np.array(reference_fixed_point_list).T)[1],
                              list(np.array(reference_fixed_point_list).T)[2],  
                              marker = 'o',
                              c = errors,
                              vmin = min_errors,
                              vmax = max_errors,
                              cmap = matplotlib.cm.hot,
                              label = 'fixed points')
      plt.colorbar(collection)
      plt.title('registration errors in mm')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      plt.show()

  return (np.mean(errors), np.std(errors), min_errors, max_errors, errors) 


def display_scalar_images(image1_z_index, image2_z_index, image1, image2, 
                          min_max_image1= (), min_max_image2 = (), title1="", title2="", figure_size=(10,8)):
    """
    Display a plot with two slices from 3D images. Display of the specific z slices is side by side.

    Note: When using this function as a callback for interact in IPython notebooks it is recommended to 
          provide the min_max_image1 and min_max_image2 variables for intensity scaling. Otherwise we
          compute them internally every time this function is invoked (scrolling events).
    Args:
        image1_z_index (int): index of the slice we display for the first image.
        image2_z_index (int): index of the slice we display for the second image.
        image1 (SimpleITK.Image): first image.
        image2 (SimpleITK.Image): second image.
        min_max_image1 (Tuple(float, float)): image intensity values are linearly scaled to be in the given range. if
                                              the range is not provided by the caller, then we use the image's minimum 
                                              and maximum intensities.
        min_max_image2 (Tuple(float, float)): image intensity values are linearly scaled to be in the given range. if
                                              the range is not provided by the caller, then we use the image's minimum 
                                              and maximum intensities.
       title1 (string): title for first image plot.
       title2 (string): title for second image plot.
       figure_size (Tuple(float,float)): width and height of figure in inches.                               
    """

    intensity_statistics_filter = sitk.StatisticsImageFilter()
    if min_max_image1:
        vmin1 = min(min_max_image1)
        vmax1 = max(min_max_image1)
    else:
        intensity_statistics_filter.Execute(image1)
        vmin1 = intensity_statistics_filter.GetMinimum()
        vmax1 = intensity_statistics_filter.GetMaximum()
    if min_max_image2:
        vmin2 = min(min_max_image2)
        vmax2 = max(min_max_image2)
    else:
        intensity_statistics_filter.Execute(image2)
        vmin2 = intensity_statistics_filter.GetMinimum()
        vmax2 = intensity_statistics_filter.GetMaximum()
    
    plt.subplots(1,2,figsize=figure_size)
    
    plt.subplot(1,2,1)
    plt.imshow(sitk.GetArrayFromImage(image1[:,:,image1_z_index]),cmap=plt.cm.Greys_r, vmin=vmin1, vmax=vmax1)
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(sitk.GetArrayFromImage(image2[:,:,image2_z_index]),cmap=plt.cm.Greys_r, vmin=vmin2, vmax=vmax2)
    plt.title(title2)
    plt.axis('off')

    plt.show()


def display_images_with_alpha(image_z, alpha, image1, image2):
    """
    Display a plot with a slice from the 3D images that is alpha blended.
    It is assumed that the two images have the same physical charecteristics (origin,
    spacing, direction, size), if they do not, an exception is thrown.
    """
    img = (1.0 - alpha)*image1[:,:,image_z] + alpha*image2[:,:,image_z] 
    plt.imshow(sitk.GetArrayFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()

def display_registration_results(fixed_image, moving_image, tx):
    moving_resampled = sitk.Resample(moving_image, fixed_image, tx, sitk.sitkLinear, 0.0, moving_image.GetPixelIDValue())
    interact(display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2]-1), alpha=(0.0,1.0,0.05), image1 = fixed(fixed_image), image2=fixed(moving_resampled));


# Callback we associate with the StartEvent, sets up our new data.
def metric_start_plot():
    global metric_values, multires_iterations
    global current_iteration_number
    
    metric_values = []
    multires_iterations = []
    current_iteration_number = -1


# Callback we associate with the EndEvent, do cleanup of data and figure.
def metric_end_plot():
    global metric_values, multires_iterations
    global current_iteration_number

    del metric_values
    del multires_iterations
    del current_iteration_number
    # Close figure, we don't want to get a duplicate of the plot latter on
    plt.close()


# Callback we associate with the IterationEvent, update our data and display 
# new figure.    
def metric_plot_values(registration_method):
    global metric_values, multires_iterations
    global current_iteration_number
    
    # Some optimizers report an iteration event for function evaluations and not
    # a complete iteration, we only want to update every iteration.
    if registration_method.GetOptimizerIteration() == current_iteration_number:
        return

    current_iteration_number =  registration_method.GetOptimizerIteration()
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot 
    # current data.
    clear_output(wait=True)
    # Plot the similarity metric values.
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()

    
# Callback we associate with the MultiResolutionIterationEvent, update the 
# index into the metric_values list. 
def metric_update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))        


# Callback we associate with the StartEvent, sets up our new data.
def metric_and_reference_start_plot():
    global metric_values, multires_iterations, reference_mean_values
    global reference_min_values, reference_max_values
    global current_iteration_number

    metric_values = []
    multires_iterations = []
    reference_mean_values = []
    reference_min_values = []
    reference_max_values = []
    current_iteration_number = -1


# Callback we associate with the EndEvent, do cleanup of data and figure.
def metric_and_reference_end_plot():
    global metric_values, multires_iterations, reference_mean_values
    global reference_min_values, reference_max_values
    global current_iteration_number
    
    del metric_values
    del multires_iterations
    del reference_mean_values
    del reference_min_values
    del reference_max_values
    del current_iteration_number
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


# Callback we associate with the IterationEvent, update our data and display 
# new figure.    
def metric_and_reference_plot_values(registration_method, fixed_points, moving_points):
    global metric_values, multires_iterations, reference_mean_values
    global reference_min_values, reference_max_values
    global current_iteration_number

    # Some optimizers report an iteration event for function evaluations and not
    # a complete iteration, we only want to update every iteration.
    if registration_method.GetOptimizerIteration() == current_iteration_number:
        return

    current_iteration_number =  registration_method.GetOptimizerIteration()
    metric_values.append(registration_method.GetMetricValue())
    # Compute and store TRE statistics (mean, min, max).
    current_transform = sitk.Transform(registration_method.GetInitialTransform())
    current_transform.SetParameters(registration_method.GetOptimizerPosition())
    current_transform.AddTransform(registration_method.GetMovingInitialTransform())
    current_transform.AddTransform(registration_method.GetFixedInitialTransform().GetInverse())
    mean_error, _, min_error, max_error, _ = registration_errors(current_transform, fixed_points, moving_points)
    reference_mean_values.append(mean_error)
    reference_min_values.append(min_error)
    reference_max_values.append(max_error)
                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data.
    clear_output(wait=True)
    # Plot the similarity metric values.
    plt.subplot(1,2,1)
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    # Plot the TRE mean value and the [min-max] range.
    plt.subplot(1,2,2)
    plt.plot(reference_mean_values, color='black', label='mean')
    plt.fill_between(range(len(reference_mean_values)), reference_min_values, reference_max_values, 
                     facecolor='red', alpha=0.5)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('TRE [mm]', fontsize=12)
    plt.legend()
    
    # Adjust the spacing between subplots so that the axis labels don't overlap.
    plt.tight_layout()
    plt.show()

body_label = 0
air_label = 1
lung_label = 2    


def read_POPI_points(file_name):
    with open(file_name,'r') as fp:
        lines = fp.readlines()
        points = []
        # First line in the file is #X Y Z which we ignore.
        for line in lines[1:]:
            coordinates = line.split()
            if coordinates:
                points.append((float(coordinates[0]), float(coordinates[1]), float(coordinates[2])))
        return points
    
    
def overlay_binary_segmentation_contours(image, mask, window_min, window_max):
    """
    Given a 2D image and mask:
       a. resample the image and mask into isotropic grid (required for display).
       b. rescale the image intensities using the given window information.
       c. overlay the contours computed from the mask onto the image.
    """
    # Resample the image (linear interpolation) and mask (nearest neighbor interpolation) into an isotropic grid, 
    # required for display.
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing, min_spacing]
    new_size = [int(round(original_size[0]*(original_spacing[0]/min_spacing))), 
                int(round(original_size[1]*(original_spacing[1]/min_spacing)))]
    resampled_img = sitk.Resample(image, new_size, sitk.Transform(), 
                                  sitk.sitkLinear, image.GetOrigin(),
                                  new_spacing, image.GetDirection(), 0.0, 
                                  image.GetPixelIDValue())
    resampled_msk = sitk.Resample(mask, new_size, sitk.Transform(), 
                                  sitk.sitkNearestNeighbor, mask.GetOrigin(),
                                  new_spacing, mask.GetDirection(), 0.0, 
                                  mask.GetPixelIDValue())

    # Create the overlay: cast the mask to expected label pixel type, and do the same for the image after
    # window-level, accounting for the high dynamic range of the CT.
    return sitk.LabelMapContourOverlay(sitk.Cast(resampled_msk, sitk.sitkLabelUInt8), 
                                       sitk.Cast(sitk.IntensityWindowing(resampled_img,
                                                                         windowMinimum=window_min, 
                                                                         windowMaximum=window_max), 
                                                 sitk.sitkUInt8), 
                                       opacity = 1, 
                                       contourThickness=[2,2])    
    
def display_coronal_with_overlay(temporal_slice, coronal_slice, images, masks, label, window_min, window_max):
    """
    Display a coronal slice from the 4D (3D+time) CT with a contour overlaid onto it. The contour is the edge of 
    the specific label.
    """
    img = images[temporal_slice][:,coronal_slice,:]
    msk = masks[temporal_slice][:,coronal_slice,:]==label

    overlay_img = overlay_binary_segmentation_contours(img, msk, window_min, window_max)    
    # Flip the image so that corresponds to correct radiological view.
    plt.imshow(np.flipud(sitk.GetArrayFromImage(overlay_img)))
    plt.axis('off')
    plt.show()

def display_coronal_with_label_maps_overlay(coronal_slice, mask_index, image, masks, label, window_min, window_max):
    """
    Display a coronal slice from a 3D CT with a contour overlaid onto it. The contour is the edge of 
    the specific label from the specific mask. Function is used to display results of transforming a segmentation
    using registration.
    """
    img = image[:,coronal_slice,:]
    msk = masks[mask_index][:,coronal_slice,:]==label

    overlay_img = overlay_binary_segmentation_contours(img, msk, window_min, window_max)
    # Flip the image so that corresponds to correct radiological view.
    plt.imshow(np.flipud(sitk.GetArrayFromImage(overlay_img)))
    plt.axis('off')
    plt.show()
