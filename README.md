# 基于Mindspore的ViT 疟疾分类  Malaria Classification via ViT based on Mindspore
本项目是数据图像处理的课程设计，利用基于Mindspore实现的ViT对疟疾厚涂片数据集进行分类，数据集包含三个类别：Falciparum，Vivax和Uninfected，可以从：[https://www.nlm.nih.gov/](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)下载。
项目的数据集和checkpoint默认保存目录不在项目工作区，如需使用请更改data_loader和model_loader中的路径

## 项目运行
项目运行在Linux上，请使用bash运行./train.py,./test.py,./predict.py，分别实现训练，测试和带有输出预测标签的图像。若使用pycharm远程解释器，运行时请通过命令： sed -i 's/r$//' xx.py 删除shebang后的Windows换行符
## 项目环境
Mindspore不能简单地通过pip命令完成配置，请参照mindspore官网完成环境配置https://www.mindspore.cn/install/
