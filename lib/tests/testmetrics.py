import numpy as np
import matplotlib.pyplot as plt
from rlxutils import subplots
from ..experiments import metrics

print ("unit testing metrics")

# -----------------------------------------
# functions to generate random data
# -----------------------------------------
def generate(m,n, intmax=10000):
    r = np.random.randint(intmax, size=(m,n))
    r = r / np.sum(r, axis=1).reshape(-1,1)
    r = np.abs(np.log(r+1e-3))
    r = np.exp(r)
    r = r / np.sum(r, axis=1).reshape(-1,1)
    return r

def generate_ytrue_ypred(n_images, size, n_classes):
    m = n_images * size**2
    y_pred = generate(m,n_classes)
    y_true = generate(m,n_classes)
    y_pred = y_pred.reshape((n_images, size, size, n_classes)).astype(np.float32)
    y_true = y_true.reshape((n_images, size, size, n_classes)).astype(np.float32)
    return y_true, y_pred


def run():
       # --------------------------------------------------------------------------------------------------
       # --------------------------------------------------------------------------------------------------
       # Check metrics with selected data vs hand calculations
       # --------------------------------------------------------------------------------------------------
       # --------------------------------------------------------------------------------------------------
       y_true = np.array([[[[0.55 , 0.237, 0.212],
              [0.149, 0.592, 0.258]],

              [[0.061, 0.456, 0.482],
              [0.752, 0.134, 0.114]]],


              [[[0.139, 0.572, 0.289],
              [0.345, 0.326, 0.329]],

              [[0.043, 0.805, 0.152],
              [0.173, 0.561, 0.266]]]])

       y_pred = np.array([[[[0.35 , 0.149, 0.501],
              [0.314, 0.488, 0.198]],

              [[0.182, 0.553, 0.265],
              [0.544, 0.298, 0.158]]],


              [[[0.429, 0.139, 0.432],
              [0.372, 0.353, 0.275]],

              [[0.507, 0.267, 0.227],
              [0.198, 0.467, 0.335]]]])

       cw = np.r_[0.1,0.6,0.3].astype(np.float32)


       y_pred_argmax = np.array([[[2, 1],
                            [1, 0]],
                            [[2, 0], 
                            [0, 1]]])

       y_true_argmax = np.array([[[0, 1],
                            [2, 0]],
                            [[1, 0],
                            [1, 1]]])

       y_pred_onehot_proportions = np.array([[0.25, 0.5 , 0.25],
                                          [0.5 , 0.25, 0.25]]).astype(np.float32)

       y_pred_proportions = np.array([[0.3475 , 0.372  , 0.2805 ],
                                   [0.3765 , 0.3065 , 0.31725]]).astype(np.float32)

       y_true_proportions = y_true.mean(axis=1).mean(axis=1).astype(np.float32)

       mse = 0.022906
       mse_onehot = 0.0424395

       mae_perclass = np.array([0.116   , 0.138375, 0.036125])
       mae_perclass_onehot = np.array([0.22649999, 0.23062499, 0.01275   ])

       mean_mae = 0.096833
       mean_mae_onehot = 0.156625

       f1 = 0.5

       # check all pixels add up to 1
       assert np.allclose(y_pred.sum(axis=-1), 1, atol=1e-3)
       assert np.allclose(y_true.sum(axis=-1), 1, atol=1e-3)

       # check computing proportions
       m = metrics.ProportionsMetrics(cw)
       assert np.allclose (m.get_y_pred_as_proportions(y_pred, argmax=False), y_pred_proportions, atol=1e-3)
       assert np.allclose (m.get_y_pred_as_proportions(y_pred, argmax=True), y_pred_onehot_proportions, atol=1e-3)

       # check mse
       assert np.allclose(m.multiclass_proportions_mse(y_true_proportions, y_pred_proportions), mse, atol=1e-5)
       assert np.allclose(m.multiclass_proportions_mse(y_true_proportions, y_pred_onehot_proportions), mse_onehot, atol=1e-5)

       # check mae per class

       assert np.allclose( mae_perclass,
                     m.multiclass_proportions_mae(y_true_proportions, y_pred_proportions, perclass=True),
                     atol=1e-5),\
              "per class mae without onehot does not match"
              
       assert np.allclose( mae_perclass_onehot,
                     m.multiclass_proportions_mae(y_true_proportions, y_pred_onehot_proportions, perclass=True),
                     atol=1e-5),\
              "per class mae with onehot does not match"

       # check mean mae
       assert np.allclose( mean_mae,
                     m.multiclass_proportions_mae(y_true_proportions, y_pred_proportions, perclass=False),
                     atol=1e-5),\
              "mean mae without onehot does not match"
              
       assert np.allclose( mean_mae_onehot,
                     m.multiclass_proportions_mae(y_true_proportions, y_pred_onehot_proportions, perclass=False),
                     atol=1e-5),\
              "mean mae with onehot does not match"

       # check f1
       cm = metrics.PixelClassificationMetrics(number_of_classes = y_pred.shape[-1])

       assert np.allclose(f1, cm.update_state(y_true_argmax, y_pred).result('f1', 'micro'), atol=1e-5), \
              "f1 micro does not match"


       # --------------------------------------------------------------------------------------------------
       # --------------------------------------------------------------------------------------------------
       # Check metrics with random data vs `numpy` implementations
       # --------------------------------------------------------------------------------------------------
       # --------------------------------------------------------------------------------------------------

       for _ in range(20):
              # ----------------------------------------------------------------------------------------------
              # generate 10 image predictions of 5x5 with 4 classes
              n_classes = 4

              y_true, y_pred = generate_ytrue_ypred(n_images=10, size=5, n_classes=n_classes)

              # generate class weights
              cw = np.random.randint(100, size=n_classes)
              cw = cw/cw.sum()

              # ----------------------------------------------------------------------------------------------
              # check all pixels add up to 1
              assert np.allclose(y_pred.sum(axis=-1), 1, atol=1e-3)
              assert np.allclose(y_true.sum(axis=-1), 1, atol=1e-3)

              # ----------------------------------------------------------------------------------------------
              # check computing of proportions for each image

              # without one_hot (no argmax)
              y_pred_proportions = y_pred.mean(axis=1).mean(axis=1)
              assert np.allclose(y_pred_proportions.sum(axis=-1), 1, atol=1e-3), "generated sample proportions without onehot do not add up to 1"

              # compute with lib
              m = metrics.ProportionsMetrics(cw)
              m_y_pred_props = m.get_y_pred_as_proportions(y_pred, argmax=False).numpy()

              # check
              assert (np.abs(m_y_pred_props - y_pred_proportions)>1e-3).sum()<2, \
                     "proportions computed without onehot encoding by 'get_y_pred_as_proportions' do not match"

              # with one_not (argmax)
              y_pred_proportions_onehot = np.r_[[(y_pred.argmax(axis=-1)==i).mean(axis=-1).mean(axis=-1) for i in range(y_pred.shape[-1])]].T.astype(np.float32)
              assert np.allclose(y_pred_proportions_onehot.sum(axis=-1), 1, atol=1e-5), "generated sample proportions with onehot do not add up to 1"

              # compute with lib
              m = metrics.ProportionsMetrics(cw)
              m_y_pred_props = m.get_y_pred_as_proportions(y_pred, argmax=True).numpy()

              # check (allow for tolerance in a few items due to tf precision)
              assert (np.abs(m_y_pred_props - y_pred_proportions_onehot)>1e-3).sum()<=2, \
                     "proportions computed with onehot encoding by 'get_y_pred_as_proportions' do not match"

              # ----------------------------------------------------------------------------------------------
              # check 'get_y_pred_as_proportions' does nothing when proportions are already given as input

              p1 = m.get_y_pred_as_proportions(y_pred_proportions, argmax=False)
              p2 = m.get_y_pred_as_proportions(y_pred_proportions, argmax=False)

              assert np.allclose(p1, p2, atol=1e-5), "'get_y_pred_as_proportions' should do nothing when provided already computed proportions"
              assert np.allclose(p1, y_pred_proportions, atol=1e-5), "'get_y_pred_as_proportions' should do nothing when provided already computed proportions"


              # ----------------------------------------------------------------------------------------------
              # check mse
              y_true_proportions = m.get_y_pred_as_proportions(y_true, argmax=False)
              mse        = ((y_true_proportions - y_pred_proportions)**2 * cw ).numpy().sum(axis=-1).mean()
              mse_onehot = ((y_true_proportions - y_pred_proportions_onehot)**2 * cw ).numpy().sum(axis=-1).mean()

              assert np.allclose(mse, m.multiclass_proportions_mse(y_true_proportions, y_pred_proportions), atol=1e-5),\
                     "mse without onehot incorrect with randomly generated data"

              assert np.allclose(mse_onehot, m.multiclass_proportions_mse(y_true_proportions, y_pred_proportions_onehot), atol=1e-5),\
                     "mse without onehot incorrect with randomly generated data"

              # ----------------------------------------------------------------------------------------------
              # check mae per class

              assert np.allclose( np.abs(y_true_proportions - y_pred_proportions).mean(axis=0),
                                   m.multiclass_proportions_mae(y_true_proportions, y_pred_proportions, perclass=True),
                                   atol=1e-5),\
                     "per class on random data mae without onehot does not match"

              assert np.allclose( np.abs(y_true_proportions - y_pred_proportions_onehot).mean(axis=0),
                                   m.multiclass_proportions_mae(y_true_proportions, y_pred_proportions_onehot, perclass=True),
                                   atol=1e-5),\
                     "per class on random data mae with onehot does not match"

              # ----------------------------------------------------------------------------------------------
              # check mean mae

              assert np.allclose( np.abs(y_true_proportions - y_pred_proportions).mean(axis=0).mean(),
                                   m.multiclass_proportions_mae(y_true_proportions, y_pred_proportions, perclass=False),
                                   atol=1e-5),\
                     "mean mae on random data without onehot does not match"

              assert np.allclose( np.abs(y_true_proportions - y_pred_proportions_onehot).mean(axis=0).mean(),
                                   m.multiclass_proportions_mae(y_true_proportions, y_pred_proportions_onehot, perclass=False),
                                   atol=1e-5),\
                     "mean mae on random data with onehot does not match"

              # ----------------------------------------------------------------------------------------------
              # check f1
              from sklearn.metrics import f1_score

              y_true_argmax[0,0,0]=2
              cm = metrics.PixelClassificationMetrics(number_of_classes = y_pred.shape[-1])
              cm.update_state(y_true.argmax(axis=-1), y_pred)

              assert np.allclose(cm.result('f1', 'micro'), 
                                   f1_score(y_true.argmax(axis=-1).reshape(-1), y_pred.argmax(axis=-1).reshape(-1), average='micro'),
                                   atol=1e-5),\
                     "f1 micro on random data does not match"
              