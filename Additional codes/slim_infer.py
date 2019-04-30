import tensorflow as tf
from model_infer import ENet_model

def main():

    graph = tf.Graph()
    with graph.as_default():

        model = ENet_model(img_height=512,
                           img_width=1024,
                           batch_size=1)

        saver = tf.train.Saver(tf.global_variables())
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, './training_logs/best_model/model_1_epoch_25.ckpt')
        saver.save(sess, './training_logs/best_model/model_1_epoch_25_final.ckpt')
        print("in=", model.imgs_ph.name)
        print("on=", model.logits.name)

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, './training_logs/best_model', 'semanticsegmentation_enet.pbtxt', as_text=True)

if __name__ == '__main__':
    main()