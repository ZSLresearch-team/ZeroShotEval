import numpy as np

import os
import os.path
import random
import sys
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class CLSWGAN:
    def __init__(self, _seenclasses, _epochCount=1, _critic_iter=1):
        """
        _epochCount (nepoch): number of epochs to train for
        _critic_iter: critic iteration, following WGAN-GP
        _seenclasses: count of seen class
        """
        self.epochCount = _epochCount
        self.critic_iter = _critic_iter
        self.seenclasses = _seenclasses

        self.g1 = tf.Graph()
        self.g2 = tf.Graph()

    def initDefoltModel(
        self, batch_size=64, resSize=2048, attSize=312, nz=312, _beta1=0.5
    ):
        """
        batch_size: input batch size
        resSize: size of visual features
        attSize: size of semantic features
        nz: size of the latent z vector
        _beta1: beta1 for adam. default=0.5
        """
        self.batch_size = batch_size
        self.resSize = resSize
        self.attSize = attSize
        self.nz = nz

        with self.g1.as_default():

            self.inputImgEmbd = tf.placeholder(
                tf.float32,
                [self.batch_size, self.resSize],
                name="input_features",
            )
            self.inputTxtEmbd = tf.placeholder(
                tf.float32,
                [self.batch_size, self.attSize],
                name="input_attributes",
            )
            self.inputNoiseImg = tf.placeholder(
                tf.float32, [self.batch_size, self.nz], name="noise"
            )
            self.inputLabel = tf.placeholder(
                tf.int32, [self.batch_size], name="input_label"
            )
            self.lambda1 = tf.placeholder(
                tf.float32, shape=[], name="gradient_penalty_regularizer"
            )
            self.cls_weight = tf.placeholder(
                tf.float32, shape=[], name="weight_classification_loss"
            )
            self.lr = tf.placeholder(
                tf.float32, shape=[], name="learning_rate_GANs"
            )
            self.beta1 = _beta1

            # model definition
            train = True
            reuse = False

            inputGenEmb = tf.concat(
                [self.inputNoiseImg, self.inputTxtEmbd], axis=1
            )

            gen_res = self.defoltGenerator(
                x=inputGenEmb, isTrainable=train, reuse=reuse
            )

            classificationLogits = self.classificationLayer(
                x=gen_res,
                classes=self.seenclasses.shape[0],
                isTrainable=False,
                reuse=reuse,
            )
            targetEmbd = tf.concat(
                [self.inputImgEmbd, self.inputTxtEmbd], axis=1
            )

            targetDisc = self.defoltDiscriminator(
                x=targetEmbd, isTrainable=train, reuse=reuse
            )
            genTargetEmbd = tf.concat([gen_res, self.inputTxtEmbd], axis=1)
            genTargetDisc = self.defoltDiscriminator(
                x=genTargetEmbd, isTrainable=train, reuse=True
            )

            # classification loss
            spSfEntr = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=classificationLogits, labels=self.inputLabel
            )
            classificationLoss = tf.reduce_mean(spSfEntr)

            # discriminator loss

            genDiscMean = tf.reduce_mean(genTargetDisc)
            targetDiscMean = tf.reduce_mean(targetDisc)
            self.discriminatorLoss = tf.reduce_mean(genTargetDisc - targetDisc)
            alpha = tf.random_uniform(
                shape=[batch_size, 1], minval=0.0, maxval=1.0
            )

            # differences = genTargetEnc - targetEnc
            # interpolates = targetEnc + (alpha*differences)
            interpolates = alpha * self.inputImgEmbd + ((1 - alpha) * gen_res)
            interpolate = tf.concat([interpolates, self.inputTxtEmbd], axis=1)

            gradients = tf.gradients(
                self.defoltDiscriminator(
                    x=interpolate, reuse=True, isTrainable=train
                ),
                [interpolates],
            )[0]

            slopes = tf.sqrt(
                tf.reduce_sum(tf.square(gradients), reduction_indices=[1])
            )
            gradientPenalty = tf.reduce_mean((slopes - 1.0) ** 2)

            gradientPenalty = self.lambda1 * gradientPenalty
            self.discriminatorLoss = self.discriminatorLoss + gradientPenalty

            # Wasserstein generator loss
            genLoss = -genDiscMean
            self.generatorLoss = genLoss + self.cls_weight * classificationLoss

            #################### getting parameters to optimize ####################
            discParams = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"
            )
            generatorParams = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"
            )

            for params in discParams:
                print(params.name)
                print("...................")

            for params in generatorParams:
                print(params.name)

            discOptimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr, beta1=self.beta1, beta2=0.999
            )

            genOptimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr, beta1=self.beta1, beta2=0.999
            )

            discGradsVars = discOptimizer.compute_gradients(
                self.discriminatorLoss, var_list=discParams
            )
            genGradsVars = genOptimizer.compute_gradients(
                self.generatorLoss, var_list=generatorParams
            )

            self.discTrain = discOptimizer.apply_gradients(discGradsVars)
            self.generatorTrain = genOptimizer.apply_gradients(genGradsVars)

            #################### what all to visualize  ############################
            tf.summary.scalar("DiscriminatorLoss", self.discriminatorLoss)
            tf.summary.scalar("ClassificationLoss", classificationLoss)
            tf.summary.scalar("GeneratorLoss", self.generatorLoss)
            tf.summary.scalar("GradientPenaltyTerm", gradientPenalty)
            tf.summary.scalar("MeanOfGeneratedImages", genDiscMean)
            tf.summary.scalar("MeanOfTargetImages", targetDiscMean)

            for g, v in discGradsVars:
                tf.summary.histogram(v.name, v)
                tf.summary.histogram(v.name + str("grad"), g)

            for g, v in genGradsVars:
                tf.summary.histogram(v.name, v)
                tf.summary.histogram(v.name + str("grad"), g)

            self.merged_all = tf.summary.merge_all()

    def train(
        self,
        iterBatch,
        ntrain,
        _lambda1=10,
        _cls_weight=1,
        _lr=0.0001,
        _beta1=0.5,
        classifier_checkpoint=49,
        logdir="./src/zeroshot_networks/gan_net/logs/",
        classifier_modeldir="./src/zeroshot_networks/gan_net/models_classifier/",
        modeldir="./src/zeroshot_networks/gan_net/models/",
    ):
        """
        Args:
            iterBatch: итератор отдающий батчи с запрошенным размером пакета
            ntrain: количество тренировочных данных
            _lambda1: gradient penalty regularizer, following WGAN-GP
            _cls_weight: weight of the classification loss
            _lr: learning rate to train GANs
            _beta1: beta1 for adam. default=0.5
            classifier_checkpoint: tells which ckpt file of tensorflow model to load
            logdir: folder to output and help print losses
            classifier_modeldir: folder to get classifier model checkpoints
            modeldir: folder to output  model checkpoints
        """
        self.modeldir = modeldir
        k = 1

        with tf.Session(graph=self.g1) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir, sess.graph)
            params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="classification"
            )

            saver = tf.train.Saver(var_list=params)

            for var in params:
                print(var.name + "\t")

            string = (
                classifier_modeldir
                + "models_"
                + str(classifier_checkpoint)
                + ".ckpt"
            )
            print(string)
            """try:
                saver.restore(sess, string)
            except:
                print("Previous weights not found of classifier") 
                sys.exit(0)
            """
            print("Model loaded")
            saver = tf.train.Saver()
            for epoch in range(self.epochCount):
                for i in range(0, ntrain, self.batch_size):
                    for j in range(self.critic_iter):
                        (
                            batch_feature,
                            batch_att,
                            batch_label,
                            z_rand,
                        ) = self.next_feed_dict(iterBatch, self.batch_size)
                        _, discLoss, merged = sess.run(
                            [
                                self.discTrain,
                                self.discriminatorLoss,
                                self.merged_all,
                            ],
                            feed_dict={
                                self.inputImgEmbd: batch_feature,
                                self.inputTxtEmbd: batch_att,
                                self.inputNoiseImg: z_rand,
                                self.inputLabel: batch_label,
                                self.lambda1: _lambda1,
                                self.cls_weight: _cls_weight,
                                self.lr: _lr,
                            },
                        )
                        print("Discriminator loss is:" + str(discLoss))
                        if j == 0:
                            summary_writer.add_summary(merged, k)

                    (
                        batch_feature,
                        batch_att,
                        batch_label,
                        z_rand,
                    ) = self.next_feed_dict(iterBatch, self.batch_size)
                    _, genLoss, merged = sess.run(
                        [
                            self.generatorTrain,
                            self.generatorLoss,
                            self.merged_all,
                        ],
                        feed_dict={
                            self.inputImgEmbd: batch_feature,
                            self.inputTxtEmbd: batch_att,
                            self.inputNoiseImg: z_rand,
                            self.inputLabel: batch_label,
                            self.lambda1: _lambda1,
                            self.cls_weight: _cls_weight,
                            self.lr: _lr,
                        },
                    )
                    print("Generator loss is:" + str(genLoss))
                    summary_writer.add_summary(merged, k)
                    k = k + 1

                saver.save(
                    sess,
                    os.path.join(
                        self.modeldir, "models_" + str(epoch) + ".ckpt"
                    ),
                )
                print("Model saved")

    def initDefoltGenerateModel(self, syn_num=100):
        """
        syn_num: number features to generate per class
        nz: size of the latent z vector
        """
        self.syn_num = syn_num

        with self.g2.as_default():
            # placeholderS :data.unseenclasses, data.attribute,
            self.syn_att = tf.placeholder(
                tf.float32,
                [self.syn_num, self.attSize],
                name="input_attributes",
            )
            self.noise_z1 = tf.placeholder(
                tf.float32, [self.syn_num, self.nz], name="noise"
            )

            # model definition
            noise1 = tf.concat([self.noise_z1, self.syn_att], axis=1)
            self.genResFeatures = self.defoltGenerator(
                noise1, isTrainable=False, reuse=False
            )

    def generateFeature(
        self, attribute, generateDir="src/zeroshot_networks/gan_net/result/"
    ):
        """
        attribute: txt attribute matrix for all classes [txtEmb.length() * nclass]
        """

        syn_res = np.empty((0, self.resSize), np.float32)
        syn_label = np.empty((0), np.float32)

        with tf.Session(graph=self.g2) as sess:

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="generator"
            )

            saver = tf.train.Saver(var_list=params)

            for var in params:
                print(var.name + "\t")
            self.modeldir = "./src/zeroshot_networks/gan_net/models/"
            string = (
                self.modeldir + "models_" + str(self.epochCount - 1) + ".ckpt"
            )
            # print (string)

            try:
                saver.restore(sess, string)
            except:
                print("Previous weights not found of generator")
                sys.exit(0)

            print("Model loaded")

            saver = tf.train.Saver()

            for i in range(0, self.seenclasses.shape[0]):
                iclass = self.seenclasses[i]
                iclass_att = np.reshape(attribute[iclass], (1, self.attSize))
                # print (iclass_att.shape)
                # print (iclass_att)
                batch_att = np.repeat(iclass_att, [self.syn_num], axis=0)
                # print (batch_att.shape)
                z_rand = np.random.normal(0, 1, [self.syn_num, self.nz]).astype(
                    np.float32
                )

                syn_features = sess.run(
                    self.genResFeatures,
                    feed_dict={self.syn_att: batch_att, self.noise_z1: z_rand},
                )

                syn_res = np.vstack((syn_res, syn_features))
                temp = np.repeat(iclass, [self.syn_num], axis=0)
                # print (temp.shape)
                syn_label = np.concatenate((syn_label, temp))

            # print (syn_res.shape)
            # print (syn_label.shape)
            np.savetxt(generateDir + "syn_res.txt", syn_res, delimiter=",")
            np.savetxt(generateDir + "syn_label.txt", syn_label, delimiter=",")

    def next_feed_dict(self, iterBatch, batch_size):
        batch_feature, batch_labels, batch_att = iterBatch.next_batch(
            batch_size
        )
        batch_label = self.map_label(batch_labels, self.seenclasses)
        z_rand = np.random.normal(0, 1, [batch_size, self.nz]).astype(
            np.float32
        )

        return batch_feature, batch_att, batch_label, z_rand

    def map_label(self, label, classes):
        mapped_label = np.empty_like(label)
        for i in range(classes.shape[0]):
            mapped_label[label == classes[i]] = i

        return mapped_label

    def defoltGenerator(
        self, x, name="generator", reuse=False, isTrainable=True
    ):

        hiddenLayerGenSize = 4096

        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            net = tf.layers.dense(
                inputs=x,
                units=hiddenLayerGenSize,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.02
                ),
                activation=tf.nn.leaky_relu,
                name="gen_fc1",
                trainable=isTrainable,
                reuse=reuse,
            )

            net = tf.layers.dense(
                inputs=net,
                units=self.resSize,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.02
                ),
                activation=tf.nn.relu,
                name="gen_fc2",
                trainable=isTrainable,
                reuse=reuse,
            )
            # the output is relu'd as the encoded representation is also the activations by relu

            return tf.reshape(net, [-1, self.resSize])

    def defoltDiscriminator(
        self, x, name="discriminator", reuse=False, isTrainable=True
    ):

        hiddenLayerDescSize = 1024
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            net = tf.layers.dense(
                inputs=x,
                units=hiddenLayerDescSize,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.02
                ),
                activation=tf.nn.leaky_relu,
                name="disc_fc1",
                trainable=isTrainable,
                reuse=reuse,
            )

            real_fake = tf.layers.dense(
                inputs=net,
                units=1,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.02
                ),
                activation=None,
                name="disc_rf",
                trainable=isTrainable,
                reuse=reuse,
            )

            return tf.reshape(real_fake, [-1])

    def classificationLayer(
        self, x, classes, name="classification", reuse=False, isTrainable=True
    ):

        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            net = tf.layers.dense(
                inputs=x,
                units=classes,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.02
                ),
                activation=None,
                name="fc1",
                trainable=isTrainable,
                reuse=reuse,
            )

            net = tf.reshape(net, [-1, classes])
        return net
