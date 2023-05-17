import tensorflow as tf
from tensorflow.keras import Model
import segmentation_models as sm

from .base import *
from ..utils.autoinit import *
from ..components.classic import *


class Custom_ConvolutionsRegression(BaseModel):
    
    def __init__(self, 
                 conv_layers, 
                 dense_layers, 
                 use_alexnet_weights=False, 
                 number_of_classes = None):
        """
        see Conv2DBlock and DenseBlock for example params
        """
        super().__init__()
        autoinit(self)

    def build(self, input_shape):
        
        self.conv_block = Conv2DBlock(self.conv_layers, start_n=1, use_alexnet_weights=self.use_alexnet_weights)
                        
        if self.dense_layers is not None:
            self.dense_block = DenseBlock(self.dense_layers, start_n=self.conv_block.end_n+1)
            self.flatten = Flatten(name=f'{self.dense_block.end_n+1:02d}_flatten')
            self.output_layer = Dense(self.number_of_classes, activation='softmax', name="probabilities")
        else:
            # if there are no dense layers, adds a conv layer with softmax with n_classes 1x1 filters so 
            # that each output pixel outputs a probability distribution. then the probability 
            # distributions of all output pixels are averaged to obtain a single output probability 
            # distribution per input image.
            self.output_layer = Conv2D(kernel_size=1, filters=self.number_of_classes, activation='softmax', strides=1, name='probabilities')            
        
    def call(self, x):
        x = self.conv_block(x)
            
        if self.dense_layers is not None:
            x = self.dense_block(x)
            x = self.flatten(x)
            x = self.output_layer(x)
        else:
            x = self.output_layer(x)
            x = tf.reduce_mean(x, axis=[1,2])

        return x
    
    def get_name(self):
        if self.dense_layers is None:
            r = f"convregr_nofc_{len(self.conv_layers)}conv"
        else:
            r = f"convregr_{len(self.conv_layers)}conv_{len(self.dense_layers)}dense"
        return r

    def produces_segmentation_probabilities(self):
        return False    
    
    def produces_label_proportions(self):
        return True
    

class KerasBackbone_ConvolutionsRegression(BaseModel):
    
    def __init__(self, 
                 number_of_classes,
                 backbone, 
                 backbone_kwargs={'weights': None},
                 dense_layers = [   
                    dict(units=1024, activation='relu'),
                    dict(units=1024, activation='relu')
                 ],
                 ):
        """
        backbone: a class under tensorflow.keras.applications
        """
        super().__init__()
        autoinit(self)

    def get_name(self):
        r = f"convregr_{self.backbone.__name__}"

        if 'weights' in self.backbone_kwargs.keys() and self.backbone_kwargs['weights'] is not None:
            r += f"_{self.backbone_kwargs['weights']}"

        return r

    def produces_segmentation_probabilities(self):
        return False       

    def produces_label_proportions(self):
        return True

    def build(self, input_shape):
        inputs       = Input(input_shape[1:])
        backbone_out = self.backbone(include_top=False, input_tensor=inputs, **self.backbone_kwargs)(inputs)
        flat         = Flatten()(backbone_out)
        dense_out    = DenseBlock(self.dense_layers, start_n=1)(flat)
        outputs      = Dense(self.number_of_classes, activation='softmax')(dense_out)
        self.model   = Model([inputs], [outputs])
     
    def call(self, x):
        return self.model(x)


class Custom_SeparatedConvolutionsRegression(BaseModel):
    """
    this class creates a separated convolutional block for each class starting from
    the same input, and the concatenates their outputs.
    if there are no dense layers an additional convolution is used to reduce the 
    output of each convolutional block to 1x1 so that each one produces its own probability.
    """
    def __init__(self, 
                 number_of_classes,
                 conv_layers,
                 dense_layers,
                 use_alexnet_weights = False
                 ):
        """
        see Conv2DBlock and DenseBlock for example args
        """
        super().__init__()
        autoinit(self)
        self.conv_layers = self.conv_layers.copy()
        self.dense_layers = self.dense_layers.copy() if self.dense_layers is not None else None
        self.use_alexnet_weights = use_alexnet_weights
    
    def get_name(self):
        if self.dense_layers is None:
            r = f"sepconvregr_nofc_{len(self.conv_layers)}conv"
        else:
            r = f"sepconvregr_{len(self.conv_layers)}conv_{len(self.dense_layers)}dense"
        return r

    def produces_segmentation_probabilities(self):
        return False       

    def produces_label_proportions(self):
        return True
    
    def build(self, input_shape):

        inputs = Input(input_shape[1:])
        
        self.conv_blocks = []
        outs = []
        
        # create a convolutional block of each class
        for i in range(self.number_of_classes):
            cb = Conv2DBlock(self.conv_layers,  name_prefix=f'{i+1:02d}_')
            out = cb(inputs)
            if self.dense_layers is None:
                # if there are no dense layers then output a single number per convolution
                out = Conv2D(kernel_size=out.shape[1:3], filters=1, activation='elu', name=f'{i+1:02d}_conv2d_to_one')(out)
            out = Flatten(name=f'{i+1:02d}_flatten')(out)

            self.conv_blocks.append(cb)
            outs.append(out)
            
        # concatenate all flattened outputs from separated convolutions
        x = tf.concat(outs, axis=-1)
            
        if self.dense_layers is None:
            outputs = tf.keras.layers.Softmax(name='softmax_output')(x)
        else:
            self.dense_block = DenseBlock(self.dense_layers, start_n=self.number_of_classes+1)
            x = self.dense_block(x)
            outputs = Dense(self.number_of_classes, activation='softmax', name="probabilities")(x)
                    
        model = Model([inputs], [outputs])      
        
        if self.use_alexnet_weights:
            print ("setting alexnet weights", flush=True)
            walex = get_alexnet_weights(kernel_size=self.conv_layers[0]['kernel_size'])
            input_convs = [i for i in model.layers if i.name.endswith('01_conv2d')]
            for input_conv in input_convs:
                w = input_conv.get_weights()
                w[0] = walex[:,:,:, :w[0].shape[-1]]
                input_conv.set_weights(w) 
                
        self.model = model

    def call(self, x):
        return self.model(x)

