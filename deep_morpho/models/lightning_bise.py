# from typing import Dict, Callable, List

# from .bise import BiSE, BiSEC
# from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning
# from ..loss import ThresholdPenalization


# class LightningBiSE(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         optimizer: Callable,
#         loss: Callable,
#         optimizer_args: Dict = {},
#         observables: [List["Observable"]] = [],
#     ):
#         super().__init__(
#             model=BiSE(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#         )
#         self.save_hyperparameters()

#     def obs_training_step(self, batch, batch_idx):
#         x, y = batch
#         predictions = self.forward(x).squeeze()

#         outputs = {}

#         loss_supervised = self.loss(predictions, y)
#         outputs['loss_supervised'] = loss_supervised
#         outputs['loss'] = loss_supervised
#         if self.do_thresh_penalization and batch_idx >= self.first_batch_pen:
#             outputs['pen_loss'] = self.pen_fn()
#             outputs['loss'] = loss_supervised + outputs['pen_loss']

#         return outputs, predictions


# class LightningBiSEC(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         optimizer: Callable,
#         output_dir: str,
#         loss: Callable,
#         do_thresh_penalization: bool = True,
#         args_thresh_penalization: Dict = {
#             'coef': .5,
#             'degree': 2,
#             'detach_weights': True,
#         },
#         first_batch_pen: int = 100,
#         optimizer_args: Dict = {},
#         observables: [List["Observable"]] = [],
#     ):
#         super().__init__(
#             model=BiSEC(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             output_dir=output_dir,
#             optimizer_args=optimizer_args,
#             observables=observables,
#         )
#         self.save_hyperparameters()

#     def obs_training_step(self, batch, batch_idx):
#         x, y = batch
#         predictions = self.forward(x).squeeze()

#         outputs = {}

#         loss_supervised = self.loss(predictions, y)
#         outputs['loss_supervised'] = loss_supervised
#         outputs['loss'] = loss_supervised
#         if self.do_thresh_penalization and batch_idx >= self.first_batch_pen:
#             outputs['pen_loss'] = self.pen_fn()
#             outputs['loss'] = loss_supervised + outputs['pen_loss']

#         return outputs, predictions
