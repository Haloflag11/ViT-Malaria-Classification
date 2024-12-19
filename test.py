#!/root/miniconda3/bin/python
from dataloader.data_loader import DataLoader
from Model.Vit import *
from mindspore import train

#checkpoint
vit_path='/root/autodl-tmp/checkpoint/vit_b_16_8-25_241.ckpt'

#Hyperparams
epoch_size = 30
momentum=0.9
num_classes=3
eval_per_epoch=1
batch_size = 16

#加载数据
train_eval_dataLoader=DataLoader()
test_data=train_eval_dataLoader.load_test()
dataset_test = datapipe(test_data,batch_size)

network=ViT(image_size=IMAGESIZE)

lr = 1
#优化器
network_opt = nn.Adam(network.trainable_params(), lr, momentum)
network_loss = CrossEntropySmooth(sparse=True,
                                  reduction="mean",
                                  smooth_factor=0.1,
                                  num_classes=num_classes)

# load ckpt
param_dict = ms.load_checkpoint(vit_path)
ms.load_param_into_net(network, param_dict)



# define metric
eval_metrics = {'Top_1_Accuracy': train.Top1CategoricalAccuracy(),
                'Top_2_Accuracy': train.TopKCategoricalAccuracy(k=2),
                'Precision':train.Precision(),
                'Recall':train.Recall(),
                'F1':train.F1(),
                'Confusion Matrix':train.ConfusionMatrix(num_classes=num_classes)}

model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=eval_metrics, amp_level="O2")


# evaluate model
result = model.eval(dataset_test)
print(result)