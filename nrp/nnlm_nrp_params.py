from exp.params import ParamSpace
import os

ps = ParamSpace()
# prefix used to identify result files

# data
default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
ps.add_value("corpus", default_corpus)
ps.add_value("ngram_size", 5)
ps.add_value("save_model", False)

# nrp params
ps.add_list("k_dim", [1000, 4000, 8000, 10000])
ps.add_list("s_active", [2, 8, 32, 64, 128])

# architecture
ps.add_value("embed_dim", 128)
ps.add_value("embed_init", "uniform")
ps.add_value("embed_init_val", 0.01)
ps.add_value("embed_share", False)

ps.add_value("h_dim", 128)
ps.add_value("h_act", "relu")
ps.add_value("num_h", 1)

ps.add_value("logit_init", "uniform")
ps.add_value("logit_init_val", 0.01)

# ps.add_value("use_f_predict", True)
ps.add_value("f_init", "uniform")
ps.add_value("f_init_val", 0.01)

ps.add_value("epochs", 100)
ps.add_value("shuffle", True)
ps.add_value("shuffle_buffer_size", 128 * 10000)
ps.add_value("batch_size", 128)

# optimizer
ps.add_value("optimizer", "ams")
ps.add_value("lr", 5e-4)
ps.add_value("optimizer_beta1", 0.9)
ps.add_value("optimizer_beta2", 0.999)
ps.add_value("optimizer_epsilon", 1e-8)
ps.add_value("lr_decay", False)

ps.add_value("early_stop", True)
ps.add_value("patience", 3)
ps.add_value("eval_threshold", 1.0)

ps.add_value("clip_grads", True)
ps.add_value("clip_local", True)
ps.add_value("clip_value", 1.0)

ps.add_value("dropout", True)
ps.add_value("keep_prob", 0.95)

ps.add_value("l2_loss", False)


print("parameter space generated")
print("number of configurations: ", ps.grid_size)

ps.write("nnlmnrp.params")
