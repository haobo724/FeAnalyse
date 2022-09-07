import numpy as np
import os,glob
def asStride(arr, sub_shape, stride):

    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]
    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs
def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):

    '''Overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.

    See also unpooling().
    '''
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]

    view = asStride(mat_pad, (f, f), stride)
    print(view.shape)


    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result

def get_min_mean(file_path):
    name = os.path.basename(file_path)
    print(name)
    npimg = np.fromfile(file_path, dtype=np.float32).reshape(1408, 1792)
    # mat = np.random.randint(0, 20, [8, 8])

    # self.dicom_array = np.moveaxis(self.dicom_array, 0, 2)
    result = poolingOverlap(npimg, 8, stride=8, method='mean', pad=False,
                            return_max_pos=False)
    min_mean = np.min(result)
    return name ,min_mean
    print(min_mean)

def start(all_file_path):
    all_file = glob.glob(all_file_path+r'\*.raw')
    print(all_file)
    saved_dic={}
    save_path =''

    with open(os.path.join(save_path,'result.txt'),'w') as f:

        for file in all_file:
            name ,value =get_min_mean(file)
            saved_dic.setdefault(name,value)
            f.write(name+':'+str(value))
            f.write('\n')
    print(saved_dic)

start(r'C:\UserData\z003nmcc\Documents\PhD\FEM-Simulation\FE Python\Binning\Breast05')