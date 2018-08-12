import tensorflow as tf

def _parse_function(filename,label):
    image_string=tf.read_file(filname)#A Tensor of type string.
    image_decode=t.image.decode_jpeg(image_string)
    image_resized=tf.image.resize_images(image_decode,[28,28])
    return image_resized,label
filenames=tf.constant(['/image1.jpg','image2.jpg'])
labels=tf.constant([0,1,2])
dataset=tf.data.Dataset.from_tensor_slices((filenames,labels))
dataset=dataset.map(_parse_function)
