import tensorflow as tf
import numpy as np
from train_Carvana.model import unet
import cv2


# 只关注训练阶段的代码
def main(argv):
    # tf.app.flags.FLAGS接受命令行传递参数或者tf.app.flags定义的默认参数
    tf_flags = tf.app.flags.FLAGS

    # gpu config.
    # tf.ConfigProto()函数用在创建session的时候，用来对session进行参数配置
    config = tf.ConfigProto()

    # tf提供了两种控制GPU资源使用的方法，第一种方式就是限制GPU的使用率:
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用50%显存
    # 第二种是让TensorFlow在运行过程中动态申请显存，需要多少就申请多少:
    # config.gpu_options.allow_growth = True

    if tf_flags.phase == "train":
        # 使用上面定义的config设置session
        with tf.Session(config=config) as sess:
            # when use queue to load data, not use with to define sess
            # 定义Unet模型
            train_model = unet.UNet(sess, tf_flags)
            # 训练Unet网络，参数：batch_size,训练迭代步......
            train_model.train(tf_flags.batch_size, tf_flags.training_steps,
                              tf_flags.summary_steps, tf_flags.checkpoint_steps, tf_flags.save_steps)
    else:
        with tf.Session(config=config) as sess:
            # test on a image pair.
            test_model = unet.UNet(sess, tf_flags)
            # test阶段:加载checkpoint文件的数据给模型参数初始化
            test_model.load(tf_flags.checkpoint)
            image, output_masks = test_model.test()
            # return numpy ndarray.

            # save two images.
            filename_A = "input.png"
            filename_B = "output_masks.png"

            cv2.imwrite(filename_A, np.uint8(image[0].clip(0., 1.) * 255.))
            cv2.imwrite(filename_B, np.uint8(output_masks[0].clip(0., 1.) * 255.))

            # Utilize cv2.imwrite() to save images.
            print("Saved files: {}, {}".format(filename_A, filename_B))


if __name__ == '__main__':
    # tf.app.flags可以定义一些默认参数，相当于接受python文件命令行执行时后面给的的参数
    tf.app.flags.DEFINE_string("output_dir", "model_output",
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("phase", "train",
                               "model phase: train/test.")
    tf.app.flags.DEFINE_string("training_set", "./datasets",
                               "dataset path for training.")
    tf.app.flags.DEFINE_string("testing_set", "./datasets/test",
                               "dataset path for testing one image pair.")
    tf.app.flags.DEFINE_integer("batch_size", 64,
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("training_steps", 100000,
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100,
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000,
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("save_steps", 500,
                                "checkpoint period.")
    tf.app.flags.DEFINE_string("checkpoint", None,
                               "checkpoint name for restoring.")
    tf.app.run(main=main)
