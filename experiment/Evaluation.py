from mindspore.train.callback import Callback

class EvaluationCallback(Callback):
    def __init__(self,model,eval_dataset,eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval
    def on_train_epoch_begin(self, run_context):
        cb_param=run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        print(f'Training Epoch: {cur_epoch}')

    def on_train_epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            print(">>>>>>>>>>>>>>>>>>eval begin<<<<<<<<<<<<<<<<<<<")
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["Evaluation Accuracy"].append(acc)
            print(f"Evaluation Accuracy:{acc} ")


