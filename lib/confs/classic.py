import tensorflow as tf

from ..models import classicregr
from ..models import classicsegm


downsampl01 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=8, filters=96, activation='relu', padding='valid', strides=4, dropout=0.1)])
            )

downsampl02 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(
                    conv_layers=[
                        dict(kernel_size=8, filters=96, activation='relu', padding='valid', strides=4, dropout=0.1)
                    ],
                    use_alexnet_weights = True
                )
            )

downsampl03 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=8, filters=8, activation='relu', padding='valid', strides=4, dropout=0.1)])
            )

downsampl04 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=8, filters=16, activation='relu', padding='valid', strides=4, dropout=0.1)])
            )

downsampl05 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=8, filters=32, activation='relu', padding='valid', strides=4, dropout=0.1)])
            )

downsampl06 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=8, filters=64, activation='relu', padding='valid', strides=4, dropout=0.1)])
            )

downsampl07 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=8, filters=128, activation='relu', padding='valid', strides=4, dropout=0.1)])
            )

# 3- best nb filters kernel size 2,4,8,16,32 stride the same
downsampl08 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=2, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)]),
                loss = 'mse',
                learning_rate = 0.001
            )

downsampl08pxce = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=2, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)]),
                loss='pxce',
                learning_rate = 0.001
            )


downsampl08a = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=2, filters=32, activation='relu', padding='valid', strides=2, dropout=0.1)]),
                loss = 'mse',
                learning_rate = 0.0001
            )


downsampl09 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=4, dropout=0.1)]),
                loss = 'mse',
                learning_rate = 0.001
            )

downsampl09pxce = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=4, dropout=0.1)]),
                loss = 'pxce',
                learning_rate = 0.001
            )

downsampl09a = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)]),
                loss = 'mse',
                learning_rate = 0.001
            )

downsampl09apxce = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)]),
                loss = 'pxce',
                learning_rate = 0.001
            )

downsampl10 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=8, filters=96, activation='relu', padding='valid', strides=8, dropout=0.1)])
            )

downsampl11 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=16, filters=96, activation='relu', padding='valid', strides=16, dropout=0.1)])
            )
downsampl12 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=32, filters=96, activation='relu', padding='valid', strides=32, dropout=0.1)])
            )

# -- 2 layers
downsampl13 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=6, filters=32, activation='relu', padding='same', strides=1, dropout=0.1),
                                                  dict(kernel_size=6, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)])
            )

downsampl14 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=6, filters=32, activation='relu', padding='same', strides=1, dropout=0.1),
                                                  dict(kernel_size=6, filters=32, activation='relu', padding='valid', strides=2, dropout=0.1)])
            )


downsampl15 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=6, filters=32, activation='relu', padding='same', strides=1, dropout=0.1),
                                                  dict(kernel_size=2, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)])
            )

downsampl16 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=6, filters=32, activation='relu', padding='same', strides=1, dropout=0.1),
                                                  dict(kernel_size=2, filters=32, activation='relu', padding='valid', strides=2, dropout=0.1)])
            )


downsampl17 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=4, filters=32, activation='relu', padding='same', strides=1, dropout=0.1),
                                                  dict(kernel_size=2, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)])
            )


downsampl18 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=6, filters=16, activation='relu', padding='same', strides=1, dropout=0.1),
                                                  dict(kernel_size=6, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1)])
            )

downsampl19 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=16, filters=96, activation='relu', padding='valid', strides=16, dropout=0.1)]),
                metrics_args = dict(mae_proportions_argmax=False)
            )

downsampl20 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(conv_layers=[dict(kernel_size=32, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)]),
                metrics_args = dict(mae_proportions_argmax=False)
            )

smvgg16 = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'vgg16'))
            )

smvgg16imgnet = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'vgg16', encoder_weights='imagenet'))
            )


unet01 = dict(
                model_class = classicsegm.Custom_UnetSegmentation,
                model_init_args = dict(nlayers = 1)
            )

unet02 = dict(
                model_class = classicsegm.Custom_UnetSegmentation,
                model_init_args = dict(nlayers = 2)
            )


unet04 = dict(
                model_class = classicsegm.Custom_UnetSegmentation,
                model_init_args = dict(nlayers = 4),
                loss='mse',
                learning_rate = 0.0001
    
            )

unet04pxce = dict(
                model_class = classicsegm.Custom_UnetSegmentation,
                model_init_args = dict(nlayers = 4),
                loss='pxce',
                learning_rate = 0.00001
    
            )

vgg16regr = dict (
                model_class = classicregr.KerasBackbone_ConvolutionsRegression,
                model_init_args = dict(number_of_classes = 5, backbone = tf.keras.applications.VGG16)
            )


convreg01 = {
           'model_class': classicregr.Custom_ConvolutionsRegression,
           'model_init_args': 
                dict(conv_layers = [
                                    dict(kernel_size=11, filters=16, activation='elu', padding='same', strides=2, dropout=0.2, maxpool=2),
                                    dict(kernel_size=11, filters=16, activation='elu', padding='same', strides=2, dropout=0.2, maxpool=2),
                                    dict(kernel_size=11, filters=16, activation='elu', padding='same', strides=4, dropout=0.2, maxpool=2),
                                    ], 
                     dense_layers = [
                                    dict(units=16, activation='relu', dropout=0.2),
                                    dict(units=16, activation='relu', dropout=0.2)
                                    ]
                    )
          }

smvgg16imgnet_mse = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'vgg16', encoder_weights='imagenet')),
                loss='mse'
            )

smvgg16imgnet_pcxe = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'vgg16', encoder_weights='imagenet')),
                loss='pxce'
            )

smvgg16_mse = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'vgg16')),
                loss='mse',
                learning_rate = 0.00001
            )

smvgg16_pcxe = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'vgg16')),
                loss='pxce'
            )

smresnet18_mse = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'resnet18')),
                loss='mse',
                learning_rate = 0.00001
            )

smresnet18_pcxe = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'resnet18')),
                loss='pxce',
                learning_rate = 0.00001
            )