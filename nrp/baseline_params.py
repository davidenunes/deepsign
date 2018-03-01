from exp.params import ParamSpace
import os

default_out_dir = os.getcwd()

ps = ParamSpace()
# prefix used to identify result files

# data
default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
ps.add_value("corpus", default_corpus)
ps.add_value("ngram_size", 4)

ps.add_value("save_model", False)
ps.add_value("out_dir", default_out_dir)

# architecture
ps.add_list("embed_dim", [64, 128, 256])
ps.add_value("embed_init", "uniform")
ps.add_value("embed_init_val", 0.01)

# number of hidden layers, hidden layer dimensions and activations
ps.add_list("h_dim", [128, 256, 512, 1024])

ps.add_list("h_act", ["elu", "tanh"])
ps.add_value("logit_init", "uniform")
ps.add_value("logit_init_val", 0.01)

ps.add_list("num_h", [1, 2])

# data pipeline
ps.add_value("epochs", 100)
ps.add_value("shuffle", True)
ps.add_value("shuffle_buffer_size", 128 * 100000)
ps.add_value("batch_size", 128)

# optimizer
ps.add_list("optimizer", ["ams"])
ps.add_list("lr", [0.001, 0.0001])
ps.add_value("optimizer_beta1", 0.9)
ps.add_value("optimizer_beta2", 0.999)
ps.add_value("optimizer_epsilon", 1e-8)

# adam and AMSGrad maintain a lr per param
ps.add_value("lr_decay", False)
# ps.add_value("lr_decay_threshold", 1e-5)

# ps.add_value("lr_decay_rate", 0.5)  # discrete decay
ps.add_value("early_stop", True)
ps.add_value("patience", 3)
ps.add_value("eval_threshold", 1.0)

ps.add_value("clip_grads", True)
ps.add_value("clip_local", True)
ps.add_value("clip_value", 1.0)

ps.add_value("dropout", True)
ps.add_value("keep_prob", 0.9)

print("parameter space generated")
print("number of configurations: ", ps.grid_size)

ps.write("baseline.params")
