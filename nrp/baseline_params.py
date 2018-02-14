from exp.params import ParamSpace

ps = ParamSpace("baseline.params")
# prefix used to identify result files

ps.add_value("corpus", "/data/datasets/ptb/")
ps.add_value("ngram_size", 4)

ps.add_value("embed_dim", 128)
ps.add_value("embed_init", "normal")
ps.add_value("embed_limits", 0.01)
ps.add_value("logit_init", "normal")
ps.add_value("logit_limits", 0.01)
ps.add_value("h_dim", 256)
ps.add_value("h_act", "relu")
ps.add_value("num_h", 1)
ps.add_value("shuffle", True)
ps.add_value("shuffle_buffer_size", 1000 * 128)

# training
ps.add_value("batch_size", 128)
ps.add_value("epochs", 4)
ps.add_value("eval_step", 0.5)

ps.add_value("learning_rate", 0.05)
ps.add_value("lr_decay", True)
ps.add_value("lr_decay_rate", 0.5)
ps.add_value("lr_decay_on_eval", True)

ps.add_value("clip_gradients", True)
ps.add_value("clip_norm", 12.0)

ps.add_value("dropout", True)
ps.add_value("keep_prob", 0.9)
