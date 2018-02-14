from exp.params import ParamSpace
import os

default_out_dir = os.getcwd()
home = os.getenv("HOME")
default_corpus = os.path.join(home, "data/datasets/ptb/")

ps = ParamSpace("baseline.params")
# prefix used to identify result files

# data
ps.add_value("corpus", default_corpus)
ps.add_value("ngram_size", 4)
ps.add_value("out_dir", default_out_dir)

# architecture
ps.add_list("embed_dim", [128, 256])
ps.add_value("embed_init", "uniform")
ps.add_value("embed_limits", 0.01)
ps.add_value("logit_init", "uniform")
ps.add_value("logit_limits", 0.01)


# number of hidden layers, hidden layer dimensions and activations
ps.add_list("num_h", [1, 2, 3])
ps.add_list("h_dim", [128, 256, 512])
ps.add_value("h_act", "relu")

# training
ps.add_value("epochs", 8)
ps.add_value("eval_step", 0.5)

ps.add_list("batch_size", [64, 128])
ps.add_value("shuffle", True)
ps.add_value("shuffle_buffer_size", 1000 * 128)

ps.add_value("learning_rate", 0.05)
ps.add_list("lr_decay", [True])
ps.add_value("lr_decay_rate", 0.5)
ps.add_value("lr_decay_on_eval", True)

ps.add_list("clip_gradients", [True, False])
ps.add_value("clip_norm", 12.0)

ps.add_list("dropout", [True, False])
ps.add_value("keep_prob", 0.9)

print(ps.grid_size)

baseline = ps
