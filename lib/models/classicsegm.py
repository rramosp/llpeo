import tensorflow as tf
from tensorflow.keras import Model
import segmentation_models as sm

from .base import *
from ..utils.autoinit import *
from ..components.classic import *


class Custom_DownsamplingSegmentation(BaseModel):

    """
    segmentation on a lower dimension image after a few convolutions
    """
    def __init__(self, 
                 conv_layers, 
                 use_alexnet_weights = False, 
                 number_of_classes = None):
        """
        see Conv2DBlock and DenseBlock for example params
        """
        super().__init__()
        autoinit(self)
                
    def build(self, input_shape):
        
        self.conv_block   = Conv2DBlock(conv_layers=self.conv_layers, start_n=1, use_alexnet_weights=self.use_alexnet_weights)
        self.output_layer = Conv2D(kernel_size=1,
                                 filters=self.number_of_classes,
                                 activation='softmax',
                                 strides=1,
                                 name='output')
        
    def call(self, x):
        x = self.conv_block(x)
        x = self.output_layer(x)
        return x

    def get_wandb_config(self, wandb_config):
        w = super().get_wandb_config(wandb_config)
        w.update({'conv_layers': self.conv_layers,
                  'use_alexnet_weights': self.use_alexnet_weights})
        return w

    def get_name(self):
        r = f"downconvsegm_{len(self.conv_layers)}conv"
        return r
    
    def produces_segmentation_probabilities(self):
        return True    

    def produces_label_proportions(self):
        return True



class Custom_UnetSegmentation(BaseModel):

    def __init__(self, nlayers = 5, 
                       activation = 'relu', 
                       initializer='he_normal', 
                       dropout = 0.1,
                       number_of_classes = None):
        super().__init__()
        autoinit(self)
        self.dropout = np.min([dropout, 0.9])

    def call(self, x):
        return self.model(x)

    def get_name(self):
        return f"custom_unet_{self.nlayers}layers"

    def build(self, input_shape):
        act = self.activation
        # Build U-Net model
        inputs = Input(input_shape[1:])

        c1 = Conv2D(16, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (inputs)
        c1 = Dropout(self.dropout) (c1)
        c1 = Conv2D(16, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)
        u9_input = p1

        if self.nlayers>=2:
            c2 = Conv2D(32, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (p1)
            c2 = Dropout(self.dropout) (c2)
            c2 = Conv2D(32, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c2)
            p2 = MaxPooling2D((2, 2)) (c2)
            u9_input = c2

            if self.nlayers>=3:
                c3 = Conv2D(64, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (p2)
                c3 = Dropout(np.min([2*self.dropout, 0.9])) (c3)
                c3 = Conv2D(64, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c3)
                p3 = MaxPooling2D((2, 2)) (c3)
                u8_input = c3

                if self.nlayers >= 4:
                    c4 = Conv2D(128, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (p3)
                    c4 = Dropout(np.min([2*self.dropout, 0.9])) (c4)
                    c4 = Conv2D(128, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c4)
                    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
                    u7_input = c4

                    if self.nlayers >= 5:
                        c5 = Conv2D(256, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (p4)
                        c5 = Dropout(np.max([3*self.dropout, 0.9])) (c5)
                        c5 = Conv2D(256, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c5)

                        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
                        u6 = concatenate([u6, c4])
                        c6 = Conv2D(128, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (u6)
                        c6 = Dropout(np.min([2*self.dropout, 0.9])) (c6)
                        c6 = Conv2D(128, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c6)
                        u7_input = c6

                    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (u7_input)
                    u7 = concatenate([u7, c3])
                    c7 = Conv2D(64, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (u7)
                    c7 = Dropout(np.min([2*self.dropout, 0.9])) (c7)
                    c7 = Conv2D(64, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c7)
                    u8_input = c7

                u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (u8_input)
                u8 = concatenate([u8, c2])
                c8 = Conv2D(32, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (u8)
                c8 = Dropout(self.dropout) (c8)
                c8 = Conv2D(32, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c8)
                u9_input = c8

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (u9_input)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (u9)
        c9 = Dropout(self.dropout) (c9)
        c9 = Conv2D(16, (3, 3), activation=act, kernel_initializer=self.initializer, padding='same') (c9)

        outputs = Conv2D(self.number_of_classes, (1, 1), activation='softmax') (c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        
    def produces_segmentation_probabilities(self):
        return True    

    def produces_label_proportions(self):
        return True


class SM_UnetSegmentation(BaseModel):

    def __init__(self, number_of_classes, sm_keywords):
        super().__init__()
        autoinit(self)
        self.backbone = self.sm_keywords['backbone_name']

    def produces_segmentation_probabilities(self):
        return True    

    def produces_label_proportions(self):
        return True

    def build(self, input_shape):
        self.unet = sm.Unet(input_shape=(None,None,3), 
                            classes = self.number_of_classes, 
                            activation = 'softmax',
                            **self.sm_keywords)

        inp = tf.keras.layers.Input(shape=(None, None, 3))
        out = self.unet(inp)
        self.model   = tf.keras.models.Model([inp], [out])
        
    def call(self, x):
        return self.model(x)

    def get_name(self):
        r = f"segmnt_{self.backbone}"

        if 'encoder_weights' in self.sm_keywords.keys() and self.sm_keywords['encoder_weights'] is not None:
            r += f"_{self.sm_keywords['encoder_weights']}"

        return r