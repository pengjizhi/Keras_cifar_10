import sys
import time
import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


# 开始下载数据集
t0 = time.time()  # 打开深度学习计时器
# CIFAR10 图片数据集
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32

X_train = X_train.astype('float32')  # uint8-->float32
X_test = X_test.astype('float32')
X_train /= 255  # 归一化到0~1区间
X_test /= 255
print('训练样例:', X_train.shape, Y_train.shape,
      ', 测试样例:', X_test.shape, Y_test.shape) # 和之前的mnist数据集不同，由于是彩色的，所以样本直接就是4维的。
#  训练样例: (50000, 32, 32, 3) (50000, 1) , 测试样例: (10000, 32, 32, 3) (10000, 1)
num_classes = 10
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
print("取数据耗时: %.2f seconds ..." % (time.time() - t0))

###################
# 1. 建立CNN模型
###################
print("开始建模CNN ...")
model = Sequential()  # 生成一个model
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', data_format='channels_last', kernel_initializer='he_normal', input_shape=X_train.shape[1:]))  # C1 卷积层

model.add(Conv2D(32, (3, 3), activation='relu'))  # C2 卷积层
model.add(MaxPooling2D(pool_size=(2, 2)))  # S3 池化
model.add(Dropout(0.25))  # 

# model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', data_format='channels_last', kernel_initializer='he_normal', input_shape=X_train.shape[1:]))  # C1 卷积层
# model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', data_format='channels_last', kernel_initializer='he_normal', input_shape=X_train.shape[1:]))  # C1 卷积层
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # S3 池化

# model.add(Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal', input_shape=X_train.shape[1:]))  # C1 卷积层
# model.add(Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal', input_shape=X_train.shape[1:]))  # C1 卷积层
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # S3 池化
model.add(Conv2D(64, (3, 3), activation='relu')) # C4

model.add(Conv2D(64, (3, 3), activation='relu')) # C5
model.add(AveragePooling2D(pool_size=(2, 2)))  # S6
model.add(Dropout(0.25))

model.add(Flatten())  # 拉平
model.add(Dense(512, activation='relu'))  # F7 全连接层, 512个神经元
model.add(Dropout(0.5))


model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))  # label为0~9共10个类别
model.summary() # 模型小结
print("建模CNN完成 ...")

###################
# 2. 训练CNN模型
###################
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
plot_model(model, to_file='model1.png', show_shapes=True)  # 画模型图

class LossHistory(keras.callbacks.Callback):
    #函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
 
    #按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        #每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        #每五秒按照当前容器里的值来绘图
        if int(time.time()) % 5 == 0:
            self.draw_p(self.losses['batch'], 'loss', 'train_batch')
            self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
            self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
            self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        if int(time.time()) % 5 == 0:
            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
            self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')
    #绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type+'_'+label+'.jpg')
    #由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
    #所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

logs_loss = LossHistory()

model.fit(X_train, Y_train, batch_size=256, epochs=500,
          validation_data=(X_test, Y_test), callbacks=[logs_loss])  # 81.34%, 224.08s
Y_pred = model.predict_proba(X_test, verbose=0)  # Keras预测概率Y_pred
print(Y_pred[:2, ])  # 取前三张图片的十类预测概率看看
score = model.evaluate(X_test, Y_test, batch_size=64,verbose=0) # 评估测试集loss损失和精度acc
print('测试集 score(val_loss): %.4f' % score[0])  # loss损失
print('测试集 accuracy: %.4f' % score[1]) # 精度acc

model.save('cifar10_trained_model.h5')
print("耗时: %.2f seconds ..." % (time.time() - t0))


logs_loss.end_draw()