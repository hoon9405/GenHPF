import math
        
# from https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
def get_slopes(n, alibi_const):
    def _get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - alibi_const)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_slopes_power_of_2(
            n
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            _get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2, alibi_const)[0::2][
                : n - closest_power_of_2
            ]
        )