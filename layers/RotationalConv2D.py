from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

import tensorflow as tf

import keras
import keras.backend as K

import numpy as np

from keras.utils.conv_utils import conv_output_length

class RotationalConv2D(keras.layers.Layer):
    def __init__(self, num_kernels, kernel_size, strides=1, padding='valid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(RotationalConv2D, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer   = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 4, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " num_filters]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_filters = input_shape[3]

        # Transform matrix
        '''
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                 self.input_num_atoms, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')
        '''
        
        output_shape = self.compute_output_shape(input_shape)
        
        self.W = self.add_weight(shape=[self.num_kernels,self.kernel_size[0],self.kernel_size[1], input_shape[-1]],
                                 initializer = self.kernel_initializer,
                                 name='W')
        
        self.b = self.add_weight(shape=[self.num_kernels],
                                 initializer = self.bias_initializer,
                                 name='b'
                                )
        
        self.built = True
    
    def call(self, input_tensor, training=None):
        
        input_shape = K.int_shape(input_tensor)
        
        patchOrig       = K.tf.extract_image_patches(input_tensor, [1, self.kernel_size[0], self.kernel_size[1], 1], [1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1], padding=self.padding.upper())
        patchOrig_shape = K.int_shape(patchOrig)
        
        # -> None, n_x, n_y, k_x*k_y*c
        
        patchesReshaped = K.reshape(patchOrig,[-1, patchOrig_shape[1], patchOrig_shape[2], self.kernel_size[0], self.kernel_size[1], input_shape[-1]])
        patchIntensity = K.sum(patchesReshaped, axis=-1)
        patchIntensity_shape = K.int_shape(patchIntensity)
        
        patchShape_int = [-1]+list(K.int_shape(input_tensor)[1:])
        patchShape = K.variable(patchShape_int, dtype='int32')

        rowMulti = K.repeat_elements(K.arange(patchIntensity_shape[3]), patchIntensity_shape[4],0)
        colMulti = K.tile(K.arange(patchIntensity_shape[4]), patchIntensity_shape[3])

        flatImages = K.reshape(patchIntensity, [-1,patchOrig_shape[1]*patchOrig_shape[2],self.kernel_size[0]*self.kernel_size[1]])

        centroidRow = K.sum(flatImages * K.cast(rowMulti,'float32'), -1) / (K.sum(flatImages, -1) + 1e-7)
        centroidCol = K.sum(flatImages * K.cast(colMulti,'float32'), -1) / (K.sum(flatImages, -1) + 1e-7)
        
        centroids = K.stack([centroidRow, centroidCol],axis=2)
        
        meanCenter = K.eval(K.stack([K.mean(K.cast(rowMulti,'float32')), K.mean(K.cast(colMulti,'float32'))]))
        centroids_rel = centroids - meanCenter

        # Rotation
        centroids_rel = K.reshape(centroids_rel,[-1,2])
        angle = K.tf.atan2(centroids_rel[:,0], centroids_rel[:,1] + 1e-7)
        
        patches_norel = K.reshape(patchOrig,[-1,self.kernel_size[0],self.kernel_size[1],input_shape[-1]])
        
        #patches_rot = K.tf.contrib.image.rotate(patches_norel, angle, interpolation='BILINEAR')
        
        def doRot(images, angles, interpolation):
            image_height = tf.cast(array_ops.shape(images)[1],
                                 dtypes.float32)[None]
            image_width = math_ops.cast(array_ops.shape(images)[2],
                                        dtypes.float32)[None]
            transformation = tf.contrib.image.angles_to_projective_transforms(angles, image_height, image_width)
            
            ### MAKE MATRIX
            
            transforms = array_ops.reshape(transformation, constant_op.constant([-1, 8]))
            num_transforms = array_ops.shape(transforms)[0]
            # Add a column of ones for the implicit last entry in the matrix.
            matrix = array_ops.reshape(
                array_ops.concat(
                    [transforms, array_ops.ones([num_transforms, 1])], axis=1),
                constant_op.constant([-1, 3, 3]))
            
            transforms = array_ops.reshape(matrix,
                                           constant_op.constant([-1, 9]))
            # Divide each matrix by the last entry (normally 1).
            transforms /= transforms[:, 8:9] + 1e-7
            transforms = transforms[:,:8]
            
            output = tf.contrib.image.transform(images,transforms,interpolation=interpolation)

            if interpolation == 'BILINEAR':
                output = K.switch(K.equal(K.tf.linalg.det(matrix), 0), doRot(images, angle, interpolation='NEAREST'), output)

            return output
        
        #patches_rot = doRot(patches_norel, angle, interpolation='NEAREST')
        patches_rot = doRot(patches_norel, angle, interpolation='BILINEAR')
        
        #patches_rot = K.tf.contrib.image.rotate(patches_norel, angle, interpolation='NEAREST')
        patches_rot = K.reshape(patches_rot, [-1] + list(patchOrig_shape[1:]))
        
        patches_rot_shape = K.int_shape(patches_rot)
        patches_rot_flat = K.reshape(patches_rot, [-1]+[1,np.product(patches_rot_shape[1:-1])]+[patches_rot_shape[-1]])
        
        # Do filtering
        w_shape = K.int_shape(self.W)
        w_flat  = K.reshape(self.W, [w_shape[0], 1, np.product(w_shape[1:])])
        
        patches_filtered = K.sum(patches_rot_flat * w_flat, axis=-1)
        patches_filtered = K.permute_dimensions(patches_filtered,[0,2,1])
        
        filtered_shape = K.int_shape(patches_filtered)
        patches_filtered = K.reshape(patches_filtered, [-1,patches_rot_shape[1],patches_rot_shape[2],filtered_shape[-1]])
        
        patches_filtered += self.b
        
        # Return
        return patches_filtered

    def ensure_validity(self):
    
        # Kernel size
        if type(self.kernel_size) == dict:
            self.kernel_size = self.kernel_size["value"]
            
        if type(self.kernel_size) == list or type(self.kernel_size) == tuple:
            self.kernel_size = np.array(self.kernel_size)
    
        if self.kernel_size.shape == ():
            self.kernel_size = self.kernel_size[()]
        
        self.kernel_size = np.array(self.kernel_size)
        
        if self.kernel_size.shape == ():
            self.kernel_size = self.kernel_size[()]
        
        try:
            self.kernel_size = np.array(self.kernel_size["value"])
        except:
            self.kernel_size = np.array(self.kernel_size)
    
    def compute_output_shape(self, input_shape):
        self.ensure_validity()
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                np.array(self.kernel_size),
                padding=self.padding,
                stride=np.array(self.strides),
                dilation=1)
            new_space.append(new_dim)

        return tuple([input_shape[0]] + new_space[0].tolist() + [self.num_kernels])

    def get_config(self):
        config = {
            'num_kernels': self.num_kernels,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            "padding":self.padding,
            "kernel_initializer":self.kernel_initializer,
            "bias_initializer":self.bias_initializer,
        }
        
        return config
        

        
class RestrictedRotationalConv2D(keras.layers.Layer):
    def __init__(self, num_kernels, kernel_size, effective_kernel_size, strides=1, padding='valid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(RestrictedRotationalConv2D, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.effective_kernel_size = np.array(effective_kernel_size)
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer   = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 4, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " num_filters]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_filters = input_shape[3]

        # Transform matrix
        '''
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                 self.input_num_atoms, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')
        '''
        
        output_shape = self.compute_output_shape(input_shape)
        
        self.W = self.add_weight(shape=[self.num_kernels,self.effective_kernel_size[0],self.effective_kernel_size[1], input_shape[-1]],
                                 initializer = self.kernel_initializer,
                                 name='W')
        
        self.b = self.add_weight(shape=[self.num_kernels],
                                 initializer = self.bias_initializer,
                                 name='b'
                                )
        
        self.built = True
    
    def call(self, input_tensor, training=None):
        
        input_shape = K.int_shape(input_tensor)
        
        patchOrig       = K.tf.extract_image_patches(input_tensor, [1, self.kernel_size[0], self.kernel_size[1], 1], [1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1], padding=self.padding.upper())
        patchOrig_shape = K.int_shape(patchOrig)
        
        # -> None, n_x, n_y, k_x*k_y*c
        
        patchesReshaped = K.reshape(patchOrig,[-1, patchOrig_shape[1], patchOrig_shape[2], self.kernel_size[0], self.kernel_size[1], input_shape[-1]])
        patchIntensity = K.sum(patchesReshaped, axis=-1)
        patchIntensity_shape = K.int_shape(patchIntensity)
        
        patchShape_int = [-1]+list(K.int_shape(input_tensor)[1:])
        patchShape = K.variable(patchShape_int, dtype='int32')

        rowMulti = K.repeat_elements(K.arange(patchIntensity_shape[3]), patchIntensity_shape[4],0)
        colMulti = K.tile(K.arange(patchIntensity_shape[4]), patchIntensity_shape[3])

        flatImages = K.reshape(patchIntensity, [-1,patchOrig_shape[1]*patchOrig_shape[2],self.kernel_size[0]*self.kernel_size[1]])

        centroidRow = K.sum(flatImages * K.cast(rowMulti,'float32'), -1) / (K.sum(flatImages, -1) + 1e-7)
        centroidCol = K.sum(flatImages * K.cast(colMulti,'float32'), -1) / (K.sum(flatImages, -1) + 1e-7)
        
        centroids = K.stack([centroidRow, centroidCol],axis=2)
        
        meanCenter = K.eval(K.stack([K.mean(K.cast(rowMulti,'float32')), K.mean(K.cast(colMulti,'float32'))]))
        centroids_rel = centroids - meanCenter

        # Rotation
        centroids_rel = K.reshape(centroids_rel,[-1,2])
        angle = K.tf.atan2(centroids_rel[:,0], centroids_rel[:,1] + 1e-7)
        
        patches_norel = K.reshape(patchOrig,[-1,self.kernel_size[0],self.kernel_size[1],input_shape[-1]])
        
        #patches_rot = K.tf.contrib.image.rotate(patches_norel, angle, interpolation='BILINEAR')
        
        def doRot(images, angles, interpolation):
            image_height = tf.cast(array_ops.shape(images)[1],
                                 dtypes.float32)[None]
            image_width = math_ops.cast(array_ops.shape(images)[2],
                                        dtypes.float32)[None]
            transformation = tf.contrib.image.angles_to_projective_transforms(angles, image_height, image_width)
            
            ### MAKE MATRIX
            
            transforms = array_ops.reshape(transformation, constant_op.constant([-1, 8]))
            num_transforms = array_ops.shape(transforms)[0]
            # Add a column of ones for the implicit last entry in the matrix.
            matrix = array_ops.reshape(
                array_ops.concat(
                    [transforms, array_ops.ones([num_transforms, 1])], axis=1),
                constant_op.constant([-1, 3, 3]))
            
            transforms = array_ops.reshape(matrix,
                                           constant_op.constant([-1, 9]))
            # Divide each matrix by the last entry (normally 1).
            transforms /= transforms[:, 8:9] + 1e-7
            transforms = transforms[:,:8]
            
            output = tf.contrib.image.transform(images,transforms,interpolation=interpolation)

            if interpolation == 'BILINEAR':
                output = K.switch(K.equal(K.tf.linalg.det(matrix), 0), doRot(images, angle, interpolation='NEAREST'), output)

            return output
        
        # Get rotated patches
        patches_rot = doRot(patches_norel, angle, interpolation='BILINEAR')
        
        # Get effective part
        padding = (self.kernel_size - self.effective_kernel_size) // 2
        ex_begin = [0]+padding.tolist()+[0]
        ex_size  = [-1]+self.effective_kernel_size.tolist()+[K.int_shape(patches_rot)[-1]]
        
        patches_rot = tf.slice(patches_rot, ex_begin, ex_size)
        
        #patches_rot = K.reshape(patches_rot, [-1] + list(patchOrig_shape[1:])) # THIS WAS ORIGINAL
        patches_rot = K.reshape(patches_rot, [-1] + list(patchOrig_shape[1:3]) + [np.product(self.effective_kernel_size) * input_shape[-1]])
        
        patches_rot_shape = K.int_shape(patches_rot)
        patches_rot_flat = K.reshape(patches_rot, [-1]+[1,np.product(patches_rot_shape[1:-1])]+[patches_rot_shape[-1]])
        
        # Do filtering
        w_shape = K.int_shape(self.W)
        w_flat  = K.reshape(self.W, [w_shape[0], 1, np.product(w_shape[1:])])
        
        patches_filtered = K.sum(patches_rot_flat * w_flat, axis=-1)
        patches_filtered = K.permute_dimensions(patches_filtered,[0,2,1])
        
        filtered_shape = K.int_shape(patches_filtered)
        patches_filtered = K.reshape(patches_filtered, [-1,patches_rot_shape[1],patches_rot_shape[2],filtered_shape[-1]])
        
        patches_filtered += self.b
        
        # Return
        return patches_filtered
    
    def ensure_validity(self):
    
        # Kernel size
        if type(self.kernel_size) == dict:
            self.kernel_size = self.kernel_size["value"]
            
        if type(self.kernel_size) == list or type(self.kernel_size) == tuple:
            self.kernel_size = np.array(self.kernel_size)
    
        if self.kernel_size.shape == ():
            self.kernel_size = self.kernel_size[()]
        
        try:
            self.kernel_size = np.array(self.kernel_size["value"])
        except:
            self.kernel_size = np.array(self.kernel_size)

        # Effective kernel size 
        if type(self.effective_kernel_size) == dict:
            self.effective_kernel_size = self.effective_kernel_size["value"]
    
        if type(self.effective_kernel_size) == list or type(self.kernel_size) == tuple:
            self.effective_kernel_size = np.array(self.effective_kernel_size)
    
        if self.effective_kernel_size.shape == ():
            self.effective_kernel_size = self.effective_kernel_size[()]
        
        try:
            self.effective_kernel_size = np.array(self.effective_kernel_size["value"])
        except:
            self.effective_kernel_size = np.array(self.effective_kernel_size)
    
    def compute_output_shape(self, input_shape):
    
        space = input_shape[1:-2]
        new_space = []
        
        self.ensure_validity()
        
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                np.array(self.kernel_size),
                padding=self.padding,
                stride=np.array(self.strides),
                dilation=1)
            new_space.append(new_dim)

        return tuple([input_shape[0]] + new_space[0].tolist() + [self.num_kernels])
        
    def get_config(self):
        config = {
            'num_kernels'           : self.num_kernels,
            'kernel_size'           : self.kernel_size,
            'effective_kernel_size' : self.effective_kernel_size,
            
            'strides'               : self.strides,
            "padding"               : self.padding,
            
            "kernel_initializer"    : self.kernel_initializer,
            "bias_initializer"      : self.bias_initializer,
        }
        
        return config

        
class RotationalConv2DPose(keras.layers.Layer):
    def __init__(self, num_kernels, kernel_size, effective_kernel_size, strides=1, padding='valid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(RotationalConv2DPose, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_size = np.array(kernel_size)
        self.effective_kernel_size = np.array(effective_kernel_size)
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer   = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 4, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " num_filters]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_filters = input_shape[3]

        # Transform matrix
        '''
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                 self.input_num_atoms, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')
        '''
        
        output_shape = self.compute_output_shape(input_shape)
        
        self.W = self.add_weight(shape=[self.num_kernels,self.effective_kernel_size[0],self.effective_kernel_size[1], input_shape[-1]],
                                 initializer = self.kernel_initializer,
                                 name='W')
        
        self.b = self.add_weight(shape=[self.num_kernels],
                                 initializer = self.bias_initializer,
                                 name='b'
                                )
        
        self.built = True
    
    def call(self, input_tensor, training=None):
        
        input_shape = K.int_shape(input_tensor)
        
        patchOrig       = K.tf.extract_image_patches(input_tensor, [1, self.kernel_size[0], self.kernel_size[1], 1], [1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1], padding=self.padding.upper())
        patchOrig_shape = K.int_shape(patchOrig)
        
        patchOrig      += K.random_uniform(K.shape(patchOrig)) * 1e-7
        
        # -> None, n_x, n_y, k_x*k_y*c
        
        patchesReshaped = K.reshape(patchOrig,[-1, patchOrig_shape[1], patchOrig_shape[2], self.kernel_size[0], self.kernel_size[1], input_shape[-1]])
        patchIntensity = K.sum(patchesReshaped, axis=-1)
        patchIntensity_shape = K.int_shape(patchIntensity)
        
        patchShape_int = [-1]+list(K.int_shape(input_tensor)[1:])
        patchShape = K.variable(patchShape_int, dtype='int32')

        rowMulti = K.repeat_elements(K.arange(patchIntensity_shape[3]), patchIntensity_shape[4],0)
        colMulti = K.tile(K.arange(patchIntensity_shape[4]), patchIntensity_shape[3])

        flatImages = K.reshape(patchIntensity, [-1,patchOrig_shape[1]*patchOrig_shape[2],self.kernel_size[0]*self.kernel_size[1]])

        centroidRow = K.sum((flatImages+1e-7) * K.cast(rowMulti,'float32'), -1) / (K.sum(flatImages, -1) + 1e-7)
        centroidCol = K.sum((flatImages+1e-7) * K.cast(colMulti,'float32'), -1) / (K.sum(flatImages, -1) + 1e-7)
        
        centroids = K.stack([centroidRow, centroidCol],axis=2)
        
        meanCenter = K.eval(K.stack([K.mean(K.cast(rowMulti,'float32')), K.mean(K.cast(colMulti,'float32'))]))
        centroids_rel = centroids - meanCenter

        # Rotation
        centroids_rel = K.reshape(centroids_rel,[-1,2])
        angle = K.tf.atan2(centroids_rel[:,0] + 1e-7, centroids_rel[:,1] + 1e-7)
        
        patches_norel = K.reshape(patchOrig,[-1,self.kernel_size[0],self.kernel_size[1],input_shape[-1]])
        
        #patches_rot = K.tf.contrib.image.rotate(patches_norel, angle, interpolation='BILINEAR')
        
        def doRot(images, angles, interpolation):
            image_height = tf.cast(array_ops.shape(images)[1],
                                 dtypes.float32)[None]
            image_width = math_ops.cast(array_ops.shape(images)[2],
                                        dtypes.float32)[None]
            transformation = tf.contrib.image.angles_to_projective_transforms(angles, image_height, image_width)
            
            ### MAKE MATRIX
            
            transforms = array_ops.reshape(transformation, constant_op.constant([-1, 8]))
            num_transforms = array_ops.shape(transforms)[0]
            # Add a column of ones for the implicit last entry in the matrix.
            matrix = array_ops.reshape(
                array_ops.concat(
                    [transforms, array_ops.ones([num_transforms, 1])], axis=1),
                constant_op.constant([-1, 3, 3]))
            
            transforms = array_ops.reshape(matrix,
                                           constant_op.constant([-1, 9]))
            # Divide each matrix by the last entry (normally 1).
            transforms /= transforms[:, 8:9] + 1e-7
            transforms = transforms[:,:8]
            
            output = tf.contrib.image.transform(images, transforms, interpolation=interpolation)

            if interpolation == 'BILINEAR':
                output = K.switch(K.equal(K.tf.linalg.det(matrix), 0), doRot(images, angle, interpolation='NEAREST'), output)
            elif interpolation == 'NEAREST':
                output = K.switch(K.equal(K.tf.linalg.det(matrix), 0), images, output)
                
            tf.check_numerics(
                output,
                "I have freaking no idea, but your transformation matrix contains NaNs. This should NOT happen!",
                name=None
            )

            return output
        
        #patches_rot = doRot(patches_norel, angle, interpolation='NEAREST')
        patches_rot = doRot(patches_norel, angle, interpolation='BILINEAR')
        
        # Get effective part
        padding = (self.kernel_size - self.effective_kernel_size) // 2
        ex_begin = [0]+padding.tolist()+[0]
        ex_size  = [-1]+self.effective_kernel_size.tolist()+[K.int_shape(patches_rot)[-1]]
        
        patches_rot = tf.slice(patches_rot, ex_begin, ex_size)
        
        #patches_rot = K.reshape(patches_rot, [-1] + list(patchOrig_shape[1:])) # THIS WAS ORIGINAL
        patches_rot = K.reshape(patches_rot, [-1] + list(patchOrig_shape[1:3]) + [np.product(self.effective_kernel_size) * input_shape[-1]])
        
        patches_rot_shape = K.int_shape(patches_rot)
        patches_rot_flat = K.reshape(patches_rot, [-1]+[1,np.product(patches_rot_shape[1:-1])]+[patches_rot_shape[-1]])
        
        # Do filtering
        w_shape = K.int_shape(self.W)
        w_flat  = K.reshape(self.W, [w_shape[0], 1, np.product(w_shape[1:])])
        
        patches_filtered = K.sum(patches_rot_flat * w_flat, axis=-1)
        patches_filtered = K.permute_dimensions(patches_filtered,[0,2,1])
        
        filtered_shape = K.int_shape(patches_filtered)
        patches_filtered = K.reshape(patches_filtered, [-1,patches_rot_shape[1],patches_rot_shape[2],filtered_shape[-1]])
        
        centroids_resh = K.reshape(centroids,[-1,2])
        
        angles_sin_reshaped = K.reshape(K.tf.sin(angle), [-1, patches_rot_shape[1], patches_rot_shape[2], 1])
        angles_cos_reshaped = K.reshape(K.tf.cos(angle), [-1, patches_rot_shape[1], patches_rot_shape[2], 1])
        
        patches_filtered += self.b
        
        patches_filtered = K.concatenate(([angles_sin_reshaped, angles_cos_reshaped, patches_filtered]))
        
        # Return
        return patches_filtered

    def ensure_validity(self):
    
        # Kernel size
        if self.kernel_size.shape == ():
            self.kernel_size = self.kernel_size[()]
        
        try:
            self.kernel_size = np.array(self.kernel_size["value"])
        except:
            self.kernel_size = np.array(self.kernel_size)

        # Effective kernel size 
        if self.effective_kernel_size.shape == ():
            self.effective_kernel_size = self.effective_kernel_size[()]
        
        try:
            self.effective_kernel_size = np.array(self.effective_kernel_size["value"])
        except:
            self.effective_kernel_size = np.array(self.effective_kernel_size)
            
    def compute_output_shape(self, input_shape):
        self.ensure_validity()
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                np.array(self.kernel_size),
                padding=self.padding,
                stride=np.array(self.strides),
                dilation=1)
            new_space.append(new_dim)

        return tuple([input_shape[0]] + new_space[0].tolist() + [2 + self.num_kernels])
    
    def get_config(self):
        config = {
            'num_kernels'           : self.num_kernels,
            'kernel_size'           : self.kernel_size,
            'effective_kernel_size' : self.effective_kernel_size,
            
            'strides'               : self.strides,
            "padding"               : self.padding,
            
            "kernel_initializer"    : self.kernel_initializer,
            "bias_initializer"      : self.bias_initializer,
        }
        
        return config
