#!/root/miniconda3/bin/python
from dataloader.data_loader import DataLoader
from Model.Vit import *
from mindspore.train import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint,EarlyStopping
from mindspore import train
from experiment.Evaluation import EvaluationCallback

#Hyperparams
epoch_size = 30
momentum=0.9
num_classes=3
eval_per_epoch=1
batch_size = 16

train_eval_dataLoader=DataLoader()
train_data,eval_data=train_eval_dataLoader.load_train_eval()
dataset_train = datapipe(train_data,batch_size)
dataset_eval = datapipe(eval_data,batch_size)

step_size=dataset_train.get_dataset_size()


network=ViT(image_size=IMAGESIZE)

lr = nn.cosine_decay_lr(min_lr=float(0.0000001),
                        max_lr=0.00005,
                        total_step=epoch_size * step_size,
                        step_per_epoch=step_size,
                        decay_epoch=30)
#优化器
network_opt = nn.Adam(network.trainable_params(), lr, momentum)
network_loss = CrossEntropySmooth(sparse=True,
                                  reduction="mean",
                                  smooth_factor=0.1,
                                  num_classes=num_classes)
# set checkpoint
ckpt_config = CheckpointConfig(save_checkpoint_steps=eval_per_epoch*step_size, keep_checkpoint_max=100)
ckpt_callback = ModelCheckpoint(prefix='vit_b_16', directory='/root/autodl-tmp/checkpoint', config=ckpt_config)

model=train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O2")

epoch_per_eval = {"epoch": [], "Evaluation Accuracy": []}
eval_cb = EvaluationCallback(model, dataset_eval, eval_per_epoch, epoch_per_eval)#验证回调函数

es_cb=EarlyStopping(monitor='loss', patience=7, verbose=True)#早停机制

if __name__ == "__main__":
    model.train(epoch_size, dataset_train, callbacks=[ckpt_callback, LossMonitor(10), eval_cb,es_cb],dataset_sink_mode=False)