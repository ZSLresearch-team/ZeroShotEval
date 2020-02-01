import os,os.path
import numpy as np
import tensorflow as tf
import sys
import random

class CLSWGAN:
    def __init__(self, _seenclasses, _epochCount=2000, _critic_iter=5):
        """
            _epochCount (nepoch): number of epochs to train for
            _critic_iter: critic iteration, following WGAN-GP
            _seenclasses: count of seen class
        """
        self.epochCount = _epochCount
        self.critic_iter = _critic_iter
        self.seenclasses = _seenclasses

        self.g1 = tf.Graph()



    def initDefoltModel(self,batch_size=64,resSize=2048,attSize=1024,nz=312):
        """
            batch_size: input batch size
            resSize: size of visual features
            attSize: size of semantic features
            nz: size of the latent z vector            
        """
        self.batch_size = batch_size
        self.resSize = resSize
        self.attSize = attSize
        self.nz = nz
        
        hiddenLayerGenSize = 4096
        hiddenLayerDescSize = 1024
        
        with self.g1.as_default():

            inputImgEmbd    = tf.placeholder(tf.float32,[batch_size, resSize],name='input_features')
            inputTxtEmbd    = tf.placeholder(tf.float32,[batch_size, attSize],name='input_attributes')
            inputNoiseImg   = tf.placeholder(tf.float32,[batch_size, nz],name='noise')
            inputLabel      = tf.placeholder(tf.int32,[batch_size],name='input_label')
            lambda1         = tf.placeholder(tf.int32,name='gradient_penalty_regularizer')
            cls_weight      = tf.placeholder(tf.int32,name='weight_classification_loss')
            lr              = tf.placeholder(tf.int32,name='learning_rate_GANs')
            beta1           = tf.placeholder(tf.int32,name='adamBeta1')
           

            # model definition
            train = True
            reuse = False

            inputGenEmb = tf.concat([inputNoiseImg, inputTxtEmbd], axis=1)
            
            gen_res = self.defoltGenerator( x=inputGenEmb,
                                            ngh=hiddenLayerGenSize,
                                            resSize=resSize,
                                            isTrainable=train,
                                            reuse=reuse)

            classificationLogits = self.classificationLayer(    x=gen_res, 
                                                                classes=self.seenclasses.shape[0],
                                                                isTrainable=False,
                                                                reuse=reuse)
            targetEmbd = tf.concat([inputImgEmbd,inputTxtEmbd], axis=1)

            targetDisc = self.defoltDiscriminator(  x=targetEmbd,
                                                    ndh=hiddenLayerDescSize,
                                                    isTrainable=train,
                                                    reuse=reuse)
            genTargetEmbd = tf.concat([gen_res,inputTxtEmbd], axis=1)
            genTargetDisc = self.defoltDiscriminator(   x=genTargetEmbd,
                                                        ndh=hiddenLayerDescSize,
                                                        isTrainable=train,
                                                        reuse=True)
            
            #classification loss
            spSfEntr = tf.nn.sparse_softmax_cross_entropy_with_logits(  logits=classificationLogits,
                                                                        labels=inputLabel)
            classificationLoss = tf.reduce_mean(spSfEntr)

            #discriminator loss
            
            genDiscMean = tf.reduce_mean(genTargetDisc)
            targetDiscMean = tf.reduce_mean(targetDisc)
            discriminatorLoss = tf.reduce_mean(genTargetDisc - targetDisc)
            alpha = tf.random_uniform(shape=[batch_size,1], minval=0.,maxval=1.)

            #differences = genTargetEnc - targetEnc
            #interpolates = targetEnc + (alpha*differences)
            interpolates = alpha * inputImgEmbd + ((1 - alpha) * gen_res)
            interpolate = tf.concat([interpolates, inputTxtEmbd], axis=1)

            gradients = tf.gradients(   self.defoltDiscriminator(   x=interpolate,
                                                                    ndh=hiddenLayerDescSize,
                                                                    reuse=True,
                                                                    isTrainable=train),
                                        [interpolates])[0]

            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradientPenalty = tf.reduce_mean((slopes-1.)**2)   
                
            gradientPenalty = lambda1*gradientPenalty
            discriminatorLoss = discriminatorLoss + gradientPenalty

            #Wasserstein generator loss
            genLoss = -genDiscMean
            generatorLoss = genLoss + cls_weight*classificationLoss

            #################### getting parameters to optimize ####################
            discParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            generatorParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            for params in discParams:
                print (params.name)
                print ('...................')

            for params in generatorParams:
                print (params.name)

            discOptimizer = tf.train.AdamOptimizer( learning_rate=lr,
                                                    beta1=beta1,
                                                    beta2=0.999)

            genOptimizer = tf.train.AdamOptimizer(  learning_rate=lr,
                                                    beta1=beta1,
                                                    beta2=0.999)

            discGradsVars = discOptimizer.compute_gradients(discriminatorLoss,var_list=discParams)    
            genGradsVars = genOptimizer.compute_gradients(generatorLoss,var_list=generatorParams)

            discTrain = discOptimizer.apply_gradients(discGradsVars)
            generatorTrain = genOptimizer.apply_gradients(genGradsVars)

            #################### what all to visualize  ############################
            tf.summary.scalar("DiscriminatorLoss",discriminatorLoss)
            tf.summary.scalar("ClassificationLoss",classificationLoss)
            tf.summary.scalar("GeneratorLoss",generatorLoss)
            tf.summary.scalar("GradientPenaltyTerm",gradientPenalty)
            tf.summary.scalar("MeanOfGeneratedImages",genDiscMean)
            tf.summary.scalar("MeanOfTargetImages",targetDiscMean)
            
            for g,v in discGradsVars:    
                tf.summary.histogram(v.name,v)
                tf.summary.histogram(v.name+str('grad'),g)
                    
            for g,v in genGradsVars:    
                tf.summary.histogram(v.name,v)
                tf.summary.histogram(v.name+str('grad'),g)

            merged_all = tf.summary.merge_all()

    def train(self, iterBatch, ntrain, _lambda1=10, _cls_weight=1,_lr=0.0001,_beta1=0.5,classifier_checkpoint=49,logdir='./logs/',classifier_modeldir='./models_classifier/',modeldir='./models/'):
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
        k=1

        with tf.Session(graph = self.g1) as sess:
            sess.run(tf.global_variables_initializer())
            

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir, sess.graph)
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classification')
                    
            saver = tf.train.Saver(var_list=params)
                    
            for var in params:
                print (var.name+"\t")

            string = classifier_modeldir+'/models_'+str(classifier_checkpoint)+'.ckpt'
            print (string) 
            try:
                saver.restore(sess, string)
            except:
                print("Previous weights not found of classifier") 
                sys.exit(0)

            print ("Model loaded")
            saver = tf.train.Saver()
            for epoch in range(self.epochCount):
                for i in range(0, ntrain, self.batch_size):
                    for j in range(self.critic_iter):
                        batch_feature,batch_att,batch_label,z_rand = next_feed_dict(iterBatch,self.batch_size)
                        _,discLoss,merged = sess.run([discTrain,discriminatorLoss,merged_all],\
                                                    feed_dict={ inputImgEmbd:batch_feature,
                                                                inputTxtEmbd:batch_att,
                                                                inputNoiseImg:z_rand,
                                                                inputLabel:batch_label,
                                                                lambda1:_lambda1,
                                                                cls_weight:_cls_weight,
                                                                lr:_lr,
                                                                beta1:_beta1})
                        print ("Discriminator loss is:"+str(discLoss))
                        if j == 0:
                            summary_writer.add_summary(merged,k)

                    batch_feature,batch_att,batch_label,z_rand = next_feed_dict(iterBatch,self.batch_size)
                    _,genLoss,merged = sess.run([generatorTrain,generatorLoss,merged_all],\
                                            feed_dict={ inputImgEmbd:batch_feature,
                                                                inputTxtEmbd:batch_att,
                                                                inputNoiseImg:z_rand,
                                                                inputLabel:batch_label,
                                                                lambda1:_lambda1,
                                                                cls_weight:_cls_weight,
                                                                lr:_lr,
                                                                beta1:_beta1})                
                    print ("Generator loss is:"+str(genLoss))
                    summary_writer.add_summary(merged, k)
                    k=k+1

                
                saver.save(sess, os.path.join(modeldir, 'models_'+str(epoch)+'.ckpt')) 
                print ("Model saved")  
  
    def next_feed_dict(self, iterBatch , batch_size):
        batch_feature, batch_labels, batch_att = iterBatch.next_batch(batch_size)
        batch_label = util.map_label(batch_labels, self.seenclasses)
        z_rand = np.random.normal(0, 1, [batch_size, self.nz]).astype(np.float32)
        
        return batch_feature,batch_att,batch_label,z_rand

    def defoltGenerator(self,x,ngh,resSize,name="generator",reuse=False,isTrainable=True):

        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            
            net = tf.layers.dense(  inputs=x, units=ngh,
                                    kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                                    activation=tf.nn.leaky_relu,
                                    name='gen_fc1',
                                    trainable=isTrainable,
                                    reuse=reuse)        
            
            net = tf.layers.dense(  inputs = net,
                                    units =resSize,
                                    kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02), \
                                    activation = tf.nn.relu,
                                    name='gen_fc2',
                                    trainable = isTrainable,
                                    reuse=reuse)
            # the output is relu'd as the encoded representation is also the activations by relu
                        
            return tf.reshape(net, [-1, resSize])

    def defoltDiscriminator(self,x,ndh,name="discriminator",reuse=False,isTrainable=True):
            
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            
            net = tf.layers.dense(  inputs=x, 
                                    units=ndh,
                                    kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                                    activation=tf.nn.leaky_relu,
                                    name='disc_fc1',
                                    trainable=isTrainable,
                                    reuse=reuse)
            
            real_fake = tf.layers.dense(    inputs=net,
                                            units=1,
                                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                                            activation=None,
                                            name='disc_rf',
                                            trainable=isTrainable,
                                            reuse=reuse)        
            
            return tf.reshape(real_fake, [-1])

    def classificationLayer(self, x,classes,name="classification",reuse=False,isTrainable=True):
       
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            net = tf.layers.dense(  inputs=x,
                                    units=classes,
                                    kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),
                                    activation=None,
                                    name='fc1',
                                    trainable=isTrainable,
                                    reuse=reuse)

            net = tf.reshape(net, [-1, classes])    
        return net
