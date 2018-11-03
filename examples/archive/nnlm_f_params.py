from exp.params import ParamSpace
import os

ps = ParamSpace()
# prefix used to identify result assets

# data
default_corpus = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")
ps.add_value("corpus", default_corpus)
ps.add_value("ngram_size", 5)
ps.add_value("save_model", False)

# nrp params

# architecture
ps.add_list("embed_dim", [128, 256])
ps.add_value("embed_init", "uniform")
ps.add_value("embed_init_val", 0.01)
ps.add_value("embed_share", False)

ps.add_list("h_dim", [256, 512])
ps.add_value("h_act", "relu")
ps.add_value("num_h", 1)

ps.add_value("logit_init", "uniform")
ps.add_value("logit_init_val", 0.01)

ps.add_value("use_f_predict", True)
ps.add_value("f_init", "uniform")
ps.add_value("f_init_val", 0.01)

ps.add_value("epochs", 100)
ps.add_value("shuffle", True)
ps.add_value("shuffle_buffer_size", 128 * 10000)
ps.add_value("batch_size", 128)

# optimizer
ps.add_value("optimizer", "sgd")

# ps.add_value("optimizer_beta1", 0.9)
# ps.add_value("optimizer_beta2", 0.999)
# ps.add_value("optimizer_epsilon", 1e-8)

ps.add_list("lr", [0.1, 0.5, 0.05])
ps.add_value("lr_decay", True)

ps.add_value("lr_decay_rate", 0.5)
ps.add_value("early_stop", True)
ps.add_value("patience", 3)
ps.add_value("eval_threshold", 1.0)

ps.add_value("clip_grads", True)
ps.add_value("clip_local", True)
ps.add_value("clip_value", 1.0)

ps.add_value("dropout", True)
ps.add_value("keep_prob", 0.95)

ps.add_value("l2_loss", False)
ps.add_value("l2_loss_coef", 1e-5)

print("parameter space generated")
print("number of configurations: ", ps.size)

ps.write("nnlm_energy.params")
