from mmcv.runner.hooks.hook import Hook
from mmcv.runner import HOOKS

@HOOKS.register_module()
class SetEpochHook(Hook):
    def before_train_epoch(self, runner):

        if hasattr(runner.model, 'module'):
            runner.model.module.epoch = runner.epoch
        else:
            runner.model.epoch = runner.epoch
