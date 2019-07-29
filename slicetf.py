import tensorflow as tf

def replace_slice(input_, replacement, begin, size=None):
    inp_shape = tf.shape(input_)
    if size is None:
        size = tf.shape(replacement)
    else:
        replacement = tf.broadcast_to(replacement, size)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)

def replace_slice_in(tensor):
    return _SliceReplacer(tensor)

class _SliceReplacer:
    def __init__(self, tensor):
        self._tensor = tensor
    def __getitem__(self, slices):
        return _SliceReplacer._Inner(self._tensor, slices)
    def with_value(self, replacement):  # Just for convenience in case you skip the indexing
        return _SliceReplacer._Inner(self._tensor, (...,)).with_value(replacement)
    class _Inner:
        def __init__(self, tensor, slices):
            self._tensor = tensor
            self._slices = slices
        def with_value(self, replacement):
            begin, size = _make_slices_begin_size(self._tensor, self._slices)
            return replace_slice(self._tensor, replacement, begin, size)

# This computes begin and size values for a set of slices
def _make_slices_begin_size(input_, slices):
    if not isinstance(slices, (tuple, list)):
        slices = (slices,)
    inp_rank = tf.rank(input_)
    inp_shape = tf.shape(input_)
    # Did we see a ellipsis already?
    before_ellipsis = True
    # Sliced dimensions
    dim_idx = []
    # Slice start points
    begins = []
    # Slice sizes
    sizes = []
    for i, s in enumerate(slices):
        if s is Ellipsis:
            if not before_ellipsis:
                raise ValueError('Cannot use more than one ellipsis in slice spec.')
            before_ellipsis = False
            continue
        if isinstance(s, slice):
            start = s.start
            stop = s.stop
            if s.step is not None:
                raise ValueError('Step value not supported.')
        else:  # Assumed to be a single integer value
            start = s
            stop = s + 1
        # Dimension this slice refers to
        i_dim = i if before_ellipsis else inp_rank - (len(slices) - i)
        dim_size = inp_shape[i_dim]
        # Default slice values
        start = start if start is not None else 0
        stop = stop if stop is not None else dim_size
        # Fix negative indices
        start = tf.cond(tf.convert_to_tensor(start >= 0), lambda: start, lambda: start + dim_size)
        stop = tf.cond(tf.convert_to_tensor(stop >= 0), lambda: stop, lambda: stop + dim_size)
        dim_idx.append([i_dim])
        begins.append(start)
        sizes.append(stop - start)
    # For empty slice specs like [...]
    if not dim_idx:
        return tf.zeros_like(inp_shape), inp_shape
    # Make full begin and size array (including omitted dimensions)
    begin_full = tf.scatter_nd(dim_idx, begins, [inp_rank])
    size_mask = tf.scatter_nd(dim_idx, tf.ones_like(sizes, dtype=tf.bool), [inp_rank])
    size_full = tf.where(size_mask,
                         tf.scatter_nd(dim_idx, sizes, [inp_rank]),
                         inp_shape)
    return begin_full, size_full
