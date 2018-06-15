import os
import random
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('file_dir',
                           './battery_word_seg/SegmentationClassRaw',
                           'folder to list all files.')

tf.app.flags.DEFINE_string('file_format', 'png', 'format.')

def ListFilesToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    print(name.split('.')[0])
                    file.write(name.split('.')[0] + "\n")
                    break

def read_examples_list(path):
  """Read list of training or validation examples.
  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
  """
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip().split(' ')[0] for line in lines]
  
def write_examples_list(path, file_lists):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  with tf.gfile.Open(path, 'w') as f:
    for label in file_lists:
      f.write('%s\n' % (label))


def main(unused_argv):
  dir=FLAGS.file_dir
  outfile=FLAGS.file_dir+"/trainval.txt"
  wildcard = FLAGS.file_format
  if os.path.exists(outfile):
      print("create please")
        
  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)
  ListFilesToTxt(dir,file,wildcard, 1)
  file.close()
  

  filenames = read_examples_list(outfile)
  # our own split.
  random.seed(42)
  random.shuffle(filenames)
  num_examples = len(filenames)
  num_train = int(0.9 * num_examples)
  train_examples = filenames[:num_train]
  print("train",len(train_examples))
  write_examples_list(FLAGS.file_dir+"/train.txt",train_examples)
  val_examples = filenames[num_train:]
  print("val",len(val_examples))
  write_examples_list(FLAGS.file_dir+"/val.txt",val_examples)
  

if __name__ == '__main__':
  tf.app.run()