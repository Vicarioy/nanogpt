import math


def scaled_dot_product_attention(query, key, value):
    """Compute softmax(query @ key.T / sqrt(d)) @ value using nested lists."""
    if not query or not key or not value:
        raise ValueError("query, key, and value must each contain at least one row")

    query_width = len(query[0])
    key_width = len(key[0])
    value_width = len(value[0])
    if query_width == 0 or key_width == 0 or value_width == 0:
        raise ValueError("query, key, and value rows must not be empty")
    if any(len(row) != query_width for row in query):
        raise ValueError("query must be rectangular")
    if any(len(row) != key_width for row in key):
        raise ValueError("key must be rectangular")
    if any(len(row) != value_width for row in value):
        raise ValueError("value must be rectangular")
    if query_width != key_width:
        raise ValueError("query and key feature widths must match")
    if len(key) != len(value):
        raise ValueError("key and value must contain the same number of rows")

    scale = math.sqrt(query_width)
    output = []
    for query_row in query:
        scores = [
            sum(
                query_item * key_item
                for query_item, key_item in zip(query_row, key_row)
            )
            / scale
            for key_row in key
        ]
        exponentials = [math.exp(score - max(scores)) for score in scores]
        denominator = sum(exponentials)
        weights = [weight / denominator for weight in exponentials]
        output.append(
            [
                sum(
                    weights[index] * value[index][column] for index in range(len(value))
                )
                for column in range(value_width)
            ]
        )

    return output


if __name__ == "__main__":
    query = [[1.0, 0.0], [0.0, 1.0]]
    key = [[1.0, 0.0], [0.0, 1.0]]
    value = [[1.0, 0.0], [0.0, 1.0]]
    expected_diagonal = 0.66976155
    expected_off_diagonal = 1.0 - expected_diagonal

    output = scaled_dot_product_attention(query, key, value)
    expected = [
        [expected_diagonal, expected_off_diagonal],
        [expected_off_diagonal, expected_diagonal],
    ]
    if any(
        abs(actual - target) > 1e-6
        for actual_row, expected_row in zip(output, expected)
        for actual, target in zip(actual_row, expected_row)
    ):
        raise AssertionError(f"Unexpected attention output: {output}")

    print("Toy attention test passed:", output)
