from tensorflow.keras import Model


class BaseModel(Model):
    
    def produces_segmentation_probabilities(self):
        raise NotImplementedError()
        
    def produces_label_proportions(self):
        raise NotImplementedError()
    
    def custom_loss(self, p, y_pred):
        raise NotImplementedError()
    
    def get_name(self):
        return self.__class__.__name__
    
    def get_additional_wandb_config(self):
        return {}
    
    def init_model_params(self, run):
        """
        invoked by Run upon run initialization, giving a chance to
        models to initialize stuff based on a run
        """
        pass

    def save_last_input_output_pair(self, x, out):
        self.last_x = x
        self.last_out = out

    def predict_segmentation(self, x):
        # by default, models outputing a segmentation mask would do it
        # directly on the call method and label proportions are computed
        # from that segmentation mask.
        # If a model outputs label proportions on the call method and 
        # produce a segmentation mask as a by product or in a different 
        # process, then it must use this method to provide it.
        # This method will regularly be invoked just after model.call,
        # so models may store intermediate results in model.call and
        # reuse them here.
        # In this default behaviour, a Run will invoke save_last_input_output_pair
        # just after model.call.

        if not self.produces_segmentation_probabilities():
            raise ValueError(f"model {self.__class__.__name__} does not produce segmentations")

        if 'last_x' in dir(self) and 'last_out' in dir(self) and x is self.last_x:
            return self.last_out
        else:
            return self(x)
