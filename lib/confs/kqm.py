from ..models import kqm

qmp01 = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=6, n_comp=16, sigma_ini=None,deep=False))                            

qmp02 = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=6, n_comp=32, sigma_ini=None,deep=False))                            

qmp03 = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=6, n_comp=64, sigma_ini=None,deep=False),
             loss = 'mse', learning_rate=0.0001)                            

qmp03s1 = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=2, n_comp=64, sigma_ini=1,deep=False),
             )

qmp03s1d = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=2, n_comp=64, sigma_ini=1,deep=True),
             )



qmp03a = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=2, pred_strides=2, n_comp=64, sigma_ini=None,deep=False))                            

qmp03b = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=4, pred_strides=4, n_comp=64, sigma_ini=None,deep=False))                            

qmp03c = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=8, pred_strides=8, n_comp=64, sigma_ini=None,deep=False))                            

qmp03d = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=16, pred_strides=16, n_comp=64, sigma_ini=None,deep=False))                            

qmp03e = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=32, pred_strides=32, n_comp=64, sigma_ini=None,deep=False))                            

qmp03f = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=6, n_comp=64, sigma_ini=None,deep=True))     

qmp03g = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=8, pred_strides=8, n_comp=64, sigma_ini=None,deep=True))                            

qmp03h = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=16, pred_strides=16, n_comp=64, sigma_ini=None,deep=True))                            



qmp04 = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=6, n_comp=128, sigma_ini=None,deep=False))                            

qmp05 = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=6, n_comp=256, sigma_ini=None,deep=False))                            

qmp06a = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=2, n_comp=64, sigma_ini=None,deep=False),
             loss = 'mse', learning_rate=0.0001)                            

qmp06aa = dict(model_class = kqm.QMPatchSegmentation,
                     model_init_args = dict(patch_size=4, pred_strides=2, n_comp=64, sigma_ini=None,deep=False),
                                  loss = 'mse', learning_rate=0.0001)


qmp06b = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=2, pred_strides=2, n_comp=64, sigma_ini=None,deep=False),
             loss = 'mse', learning_rate=0.0001)                            

qmp06c = dict(model_class = kqm.QMPatchSegmentation,
             model_init_args = dict(patch_size=6, pred_strides=6, n_comp=64, sigma_ini=None,deep=True),
             loss = 'mse', learning_rate=0.0001)                            





aeqm = dict(
            model_class = kqm.AEQMPatchSegm,
            model_init_args = dict(
                        patch_size=6,
                        pred_strides=2,
                        n_comp=64, 
                        sigma_ini=None
                    )
        )


qmr01 = dict( 
            model_class = kqm.QMRegression,
            model_init_args = dict(n_comp = 64,sigma_ini = None)
            )
