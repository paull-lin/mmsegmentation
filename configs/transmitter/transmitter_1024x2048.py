_base_ = [
    '../_base_/models/transmitter_base.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

crop_size = (1024, 2048)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
