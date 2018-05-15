from tensorflow.python.ops import array_ops, math_ops, variables, candidate_sampling_ops, sparse_ops
from tensorflow.python.framework import dtypes, ops, sparse_tensor
from tensorflow.python.ops.nn import embedding_lookup_sparse, embedding_lookup, sigmoid_cross_entropy_with_logits
import tensorx as tx
from deepsign.rp.tf_utils import RandomIndexTensor


def _sum_rows(x):
    with ops.name_scope("row_sum"):
        """Returns a vector summing up each row of the matrix x."""
        # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
        # a matrix.  The gradient of _sum_rows(x) is more efficient than
        # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
        # we use _sum_rows(x) in the nce_loss() computation since the loss
        # is mostly used for training.
        cols = array_ops.shape(x)[1]
        ones_shape = array_ops.stack([cols, 1])
        ones = array_ops.ones(ones_shape, x.dtype)
        return array_ops.reshape(math_ops.matmul(x, ones), [-1])


# TODO we can simply generate a set of randomly generated random
# indexes to be used as negative examples, it would work a bit like
# using uniform random sample generator
# the difference is that we are not sampling from random indexes in
# existence (according to the dataset that is begin processed)
# but from the huge space of possible random indices (is this actually useful)

# TODO the problem is that we cant sample the biases associated with
# these sampled ris, we can simply ignore the biases

def generate_ri(k, s, n):
    """ Generates n random indexes with k dimension and s active units

    Args:
        k: dimension for random indexes
        s: an even number s << k
        n: number of ris to be generated

    Returns:

    """
    s = s // 2 * 2
    density = s / k

    ris = tx.sparse_random_mask([n, k],
                                density=density,
                                mask_values=[1, -1],
                                symmetrical=True)

    return ris


def sample_ri(k, s, n, true_ri):
    """ Sample Random Index

    Args:
        k: dimension of random index
        s: number of active units in the random index
        n: number of random indices to be sampled
        true_ri: sparse tensor with true ri indices

    Returns:
        (SparseTensor, float32, float32) with the sampled sparse indices,
        the expected probability of the true random index and the expected
        probability of each sparse index

        Since each random index is composed of multiple active indices, the
        expected count is computed as the multiplication of the expectations
        for each indice
    """
    flat_ri_indices = true_ri.indices[:, 1]
    true_ri_indices = array_ops.reshape(flat_ri_indices, [-1, s])

    sampled_ids, true_expected, sampled_expected = tx.sample_with_expected(k, s, true_ri_indices, num_true=s,
                                                                           batch_size=n, unique=True)
    sampled_ri_indices = tx.column_indices_to_matrix_indices(sampled_ids, dtype=dtypes.int64)

    ri_values = array_ops.tile([1., -1.], [n * (s // 2)])

    sp = sparse_tensor.SparseTensor(sampled_ri_indices, ri_values, [n, k])
    sp = sparse_ops.sparse_reorder(sp)

    true_expected = math_ops.reduce_prod(true_expected, axis=-1)
    # expected expected prob for the batch of ris
    true_expected = math_ops.reduce_mean(true_expected, axis=0)
    true_expected = array_ops.expand_dims(true_expected, axis=-1)
    sampled_expected = math_ops.reduce_prod(sampled_expected, axis=-1)

    return sp, true_expected, sampled_expected


def _compute_random_ri_sampled_logits(ri_tensors,
                                      k_dim,
                                      s_active,
                                      weights,
                                      labels,
                                      inputs,
                                      num_sampled,
                                      num_true=1,
                                      subtract_log_q=True,
                                      partition_strategy="mod",
                                      name=None,
                                      seed=None):
    """ Random Random Index Sampled Logits with negative sampling

    https://arxiv.org/pdf/1410.8251.pdf

    Computes the sampled logits from the space of all possible random indexes.
    Since any random index is possible, we sample, not from the existing random indexes
    but from the space of possible random indexes so that the model learns which combinations
    of bases are NOT the ones used to predict a given feature.

    Args:
        ri_tensors:
        k_dim:
        s_active:
        weights:
        labels:
        inputs:
        num_sampled:
        sampled_values:
        num_true:
        subtract_log_q:
        remove_accidental_hits:
        partition_strategy:
        name:
        seed:

    Returns:

    """
    if isinstance(weights, variables.PartitionedVariable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    with ops.name_scope(name, "random_ri_sampled_logits",
                        weights + [inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        true_ris = tx.gather_sparse(sp_tensor=ri_tensors, ids=labels_flat)
        sampled_ris, expected_true_ris, expected_sampled_ris = sample_ri(k_dim, s_active, num_sampled, true_ris)

        all_ris = sparse_ops.sparse_concat(axis=0, sp_inputs=[true_ris, sampled_ris])

        sp_values = all_ris
        sp_indices = tx.sparse_indices(sp_values)

        # Retrieve the weights

        # weights shape is [num_classes, dim]
        all_w = embedding_lookup_sparse(
            weights, sp_indices, sp_values, combiner="sum", partition_strategy=partition_strategy)

        # true_w shape is [batch_size * num_true, dim]
        true_w = array_ops.slice(all_w, [0, 0],
                                 array_ops.stack(
                                     [array_ops.shape(labels_flat)[0], -1]))

        sampled_w = array_ops.slice(
            all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # Apply X*W', which yields [batch_size, num_sampled]
        sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

        dim = array_ops.shape(true_w)[1:2]
        new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
        row_wise_dots = math_ops.multiply(
            array_ops.expand_dims(inputs, 1),
            array_ops.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,
                                           array_ops.concat([[-1], dim], 0))
        true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that label appears in sampled.
            true_logits -= math_ops.log(expected_true_ris)
            sampled_logits -= math_ops.log(expected_sampled_ris)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = array_ops.concat([true_logits, sampled_logits], 1)

        # true_logits is a float tensor, ones_like(true_logits) is a float
        # tensor of ones. We then divide by num_true to ensure the per-example
        # labels sum to 1.0, i.e. form a proper probability distribution.
        out_labels = array_ops.concat([
            array_ops.ones_like(true_logits) / num_true,
            array_ops.zeros_like(sampled_logits)
        ], 1)

        return out_logits, out_labels


def _compute_ri_sampled_logits(ri_tensors,
                               weights,
                               labels,
                               inputs,
                               num_sampled,
                               num_classes,
                               sampled_values,
                               num_true=1,
                               subtract_log_q=True,
                               remove_accidental_hits=False,
                               partition_strategy="mod",
                               name=None,
                               seed=None):
    if isinstance(weights, variables.PartitionedVariable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    with ops.name_scope(name, "ri_sampled_logits",
                        weights + [inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        if sampled_values is None:
            sampled_values = candidate_sampling_ops.uniform_candidate_sampler(
                true_classes=labels,
                num_true=num_true,
                num_sampled=num_sampled,
                unique=True,
                range_max=num_classes,
                seed=seed)
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence
        sampled, true_expected_count, sampled_expected_count = (
            array_ops.stop_gradient(s) for s in sampled_values)
        # pylint: enable=unpacking-non-sequence
        sampled = math_ops.cast(sampled, dtypes.int64)

        all_ids = array_ops.concat([labels_flat, sampled], 0)

        # true_ris = tx.gather_sparse(ri_tensors, labels_flat)
        # another way is to sample from ri_tensor
        # sampled_ris = generate_ri(k, s, num_sampled)
        # all_ris = sparse_ops.sparse_concat(0, [true_ris, sampled_ris])

        all_ris = tx.gather_sparse(sp_tensor=ri_tensors, ids=all_ids)
        sp_values = all_ris
        sp_indices = tx.sparse_indices(sp_values)

        # Retrieve the true weights and the logits of the sampled weights.

        # weights shape is [num_classes, dim]
        all_w = embedding_lookup_sparse(
            weights, sp_indices, sp_values, combiner="sum", partition_strategy=partition_strategy)

        # true_w shape is [batch_size * num_true, dim]
        true_w = array_ops.slice(all_w, [0, 0],
                                 array_ops.stack(
                                     [array_ops.shape(labels_flat)[0], -1]))

        sampled_w = array_ops.slice(
            all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # Apply X*W', which yields [batch_size, num_sampled]inputs
        # for energy based models the inputs are the predicted feature vectors
        sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = array_ops.shape(true_w)[1:2]
        new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
        row_wise_dots = math_ops.multiply(
            array_ops.expand_dims(inputs, 1),
            array_ops.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,
                                           array_ops.concat([[-1], dim], 0))
        true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])

        if remove_accidental_hits:
            acc_hits = candidate_sampling_ops.compute_accidental_hits(
                labels, sampled, num_true=num_true)
            acc_indices, acc_ids, acc_weights = acc_hits

            # This is how SparseToDense expects the indices.
            acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
            acc_ids_2d_int32 = array_ops.reshape(
                math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
            sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                              "sparse_indices")
            # Create sampled_logits_shape = [batch_size, num_sampled]
            sampled_logits_shape = array_ops.concat(
                [array_ops.shape(labels)[:1],
                 array_ops.expand_dims(num_sampled, 0)], 0)
            if sampled_logits.dtype != acc_weights.dtype:
                acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
            sampled_logits += sparse_ops.sparse_to_dense(
                sparse_indices,
                sampled_logits_shape,
                acc_weights,
                default_value=0.0,
                validate_indices=False)

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that label appears in sampled.
            true_logits -= math_ops.log(true_expected_count)
            sampled_logits -= math_ops.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = array_ops.concat([true_logits, sampled_logits], 1)

        # true_logits is a float tensor, ones_like(true_logits) is a float
        # tensor of ones. We then divide by num_true to ensure the per-example
        # labels sum to 1.0, i.e. form a proper probability distribution.
        out_labels = array_ops.concat([
            array_ops.ones_like(true_logits) / num_true,
            array_ops.zeros_like(sampled_logits)
        ], 1)

        return out_logits, out_labels


def ri_nce_loss(ri_tensors,
                weights,
                labels,
                inputs,
                num_sampled,
                num_classes,
                num_true=1,
                sampled_values=None,
                remove_accidental_hits=False,
                partition_strategy="mod",
                name="nce_loss"):
    with ops.name_scope(name):
        logits, labels = _compute_ri_sampled_logits(
            ri_tensors=ri_tensors,
            weights=weights,
            labels=labels,
            inputs=inputs,
            num_sampled=num_sampled,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_values,
            subtract_log_q=True,
            remove_accidental_hits=remove_accidental_hits,
            partition_strategy=partition_strategy,
            name=name)
        sampled_losses = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name="sampled_losses")
        # sampled_losses is batch_size x {true_loss, sampled_losses...}
        # We sum out true and sampled losses.

        # TODO I was getting NAN VALUES HERE, reducing the lr solved it but I must investigate the cause, sigmoid
        # cross entropy returns NaNs
        return _sum_rows(sampled_losses)
        # return sampled_losses
        #return math_ops.reduce_sum(sampled_losses, ax is=-1)


def random_ri_nce_loss(ri_tensors,
                       k_dim,
                       s_active,
                       weights,
                       labels,
                       inputs,
                       num_sampled,
                       num_classes,
                       num_true=1,
                       partition_strategy="mod",
                       name="nce_loss"):
    with ops.name_scope(name):
        logits, labels = _compute_random_ri_sampled_logits(
            ri_tensors=ri_tensors,
            k_dim=k_dim,
            s_active=s_active,
            weights=weights,
            labels=labels,
            inputs=inputs,
            num_sampled=num_sampled,
            num_true=num_true,
            subtract_log_q=True,
            partition_strategy=partition_strategy,
            name=name)
        sampled_losses = tx.binary_cross_entropy(
            labels=labels, logits=logits, name="sampled_losses")
        # sampled_losses is batch_size x {true_loss, sampled_losses...}
        # We sum out true and sampled losses.

        return _sum_rows(sampled_losses)
