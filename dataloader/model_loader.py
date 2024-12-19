
from Model.Vit import ViT, IMAGESIZE, CrossEntropySmooth
from mindspore import train,nn
import mindspore as ms
#checkpoint

class ModelLoader():
    def __init__(self,eval_metrics= {'Top_1_Accuracy': train.Top1CategoricalAccuracy(),
                'Top_2_Accuracy': train.TopKCategoricalAccuracy(k=2),
                'Precision':train.Precision(),
                'Recall':train.Recall(),
                'F1':train.F1(),
                'Confusion Matrix':train.ConfusionMatrix(num_classes=3)}
                 ,
                 vit_path='/root/autodl-tmp/checkpoint/vit_b_16_9-27_241.ckpt'



                 ):
        super().__init__()
        self.eval_metrics = eval_metrics
        self.vit_path=vit_path

    def model_loader(self):
        #Hyperparams
        epoch_size = 30
        momentum=0.9
        num_classes=3
        eval_per_epoch=1
        batch_size = 16

        #加载数据


        network=ViT(image_size=IMAGESIZE)

        lr = 1
        #优化器
        network_opt = nn.Adam(network.trainable_params(), lr, momentum)
        network_loss = CrossEntropySmooth(sparse=True,
                                          reduction="mean",
                                          smooth_factor=0.1,
                                          num_classes=num_classes)

        # load ckpt
        param_dict = ms.load_checkpoint(self.vit_path)
        ms.load_param_into_net(network, param_dict)



        # define metric


        model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=self.eval_metrics, amp_level="O2")
        return model