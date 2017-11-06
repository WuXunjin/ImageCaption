# ImageCaption
## 网络架构: 改进LSTM + VGG-16

- classifier文件夹里的rnn.py为整个网络的核心程序，涉及前向传播、计算损失和梯度、方向传播以及最后的测试程序；
- Datasets为数据集，使用的是coco的数据集；
- captioning_solver.py主要为一个训练的大框架，涉及采用什么优化方法，设定mini-batch的大小、各种超参数的值，以及打印一些训练中的数据；
- coco_utils.py为初始化coco数据集的样本；如初始化图片的卷积特征（VGG16分类层之前的输出特征），初始化图片的描述，以及对应每张图片的URL等；
- image_caption.py为外部的调用程序，负责调用整个网络；
- image_utils.py从每张图片所对应的URL下载对应图片；
- optim.py为一些最优化方法，有SGD、Momentum、Adam等；
- rnn_layers.py中包含rnn/lstm的每一“层”的程序，以及将每一层组合在一起的完整rnn/lstm程序，修改sigmoid/softmax关于每个时刻的版本。

## 主要原理：
- 将CNN提取的图像卷积特征作为LSTM隐藏层的初始状态（这里涉及到将图像特征映射到隐藏层状态的参数W_proj/b_proj，是需要网络自己学习的）;
- LSTM/RNN的输入x则选择为每个图像的文字描述所对应的word2vec向量（涉及到将图像描述中的单词转化为能够作为LSTM/RNN输入词向量的参数W_embed/b_embed，同样是需要网络去学习的），
- 再将LSTM/RNN输出层的隐藏状态h用来计算任一样本在每个时刻下对词典中每个单词的得分情况（涉及到将LSTM/RNN的输出转化为单词得分情况的参数W_vocab/b_vocab，需要网络去学习）。
- 利用SoftMax对上述得分情况计算损失和梯度，最后进行反向传播！  

![原理框图.png](http://upload-images.jianshu.io/upload_images/3970488-93887b61d610a314.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  

## 改进RNN/LSTM：
虽然RNN/LSTM当前时间的状态与之前所有时间的状态都有关，但是当时间较长时，整个网络对前面状态的依赖性会变弱，所以这里为了让图片的描述效果更好，我们在每个时隙都给RNN/LSTM的隐层状态加上图片的卷积提取特征，可以使得整个网络在任何时刻下都可以很清楚的“记住”图片的特征，并能给出更恰当的描述，因此描述的效果会更加出色；

**实验结果**如下（下面两幅图中，‘val’表示交叉验证集，第二行表示网络对于当前图片预测的描述，第三行表示图像原本的描述‘GT：’）：

![Fig1.png](http://upload-images.jianshu.io/upload_images/3970488-3abcafe945ec489d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
