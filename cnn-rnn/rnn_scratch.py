import torch

batch_size = 32
seq_length = 10
input_size = 8
hidden_size = 16
output_size = 4
learning_rate = 0.1
max_grad_norm = 1.0


def forward(inputs, parameters):
    """Return per-step logits and the values needed by BPTT."""
    hidden = inputs.new_zeros(inputs.size(0), hidden_size)
    hidden_states = []
    previous_states = []
    logits_by_step = []

    for step in range(inputs.size(1)):
        previous_states.append(hidden)
        hidden = torch.tanh(
            inputs[:, step, :] @ parameters["W_xh"]
            + hidden @ parameters["W_hh"]
            + parameters["b_h"]
        )
        hidden_states.append(hidden)
        logits_by_step.append(hidden @ parameters["W_hy"] + parameters["b_y"])

    return torch.stack(logits_by_step, dim=1), hidden_states, previous_states


def bptt(inputs, targets, parameters, hidden_states, previous_states, logits):
    """Accumulate the loss gradients while traversing the sequence backwards."""
    gradients = {name: torch.zeros_like(value) for name, value in parameters.items()}
    batch, steps = targets.shape
    next_hidden_gradient = torch.zeros_like(hidden_states[0])

    with torch.no_grad():
        probability_gradient = torch.softmax(logits, dim=-1)
        probability_gradient -= torch.nn.functional.one_hot(
            targets, num_classes=output_size
        ).to(logits.dtype)
        probability_gradient /= batch * steps

        for step in reversed(range(steps)):
            step_gradient = probability_gradient[:, step, :]
            gradients["W_hy"] += hidden_states[step].T @ step_gradient
            gradients["b_y"] += step_gradient.sum(dim=0, keepdim=True)

            hidden_gradient = (
                step_gradient @ parameters["W_hy"].T + next_hidden_gradient
            )
            preactivation_gradient = hidden_gradient * (
                1 - hidden_states[step].square()
            )
            gradients["W_xh"] += inputs[:, step, :].T @ preactivation_gradient
            gradients["W_hh"] += previous_states[step].T @ preactivation_gradient
            gradients["b_h"] += preactivation_gradient.sum(dim=0, keepdim=True)
            next_hidden_gradient = preactivation_gradient @ parameters["W_hh"].T

    return gradients


def main():
    torch.manual_seed(42)
    scale = 0.01
    parameters = {
        "W_xh": torch.randn(input_size, hidden_size) * scale,
        "W_hh": torch.randn(hidden_size, hidden_size) * scale,
        "b_h": torch.zeros(1, hidden_size),
        "W_hy": torch.randn(hidden_size, output_size) * scale,
        "b_y": torch.zeros(1, output_size),
    }
    for parameter in parameters.values():
        parameter.requires_grad_()

    inputs = torch.randn(batch_size, seq_length, input_size)
    targets = torch.randint(output_size, (batch_size, seq_length))

    logits, hidden_states, previous_states = forward(inputs, parameters)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, output_size), targets.reshape(-1)
    )
    loss.backward()

    gradients = bptt(
        inputs, targets, parameters, hidden_states, previous_states, logits
    )
    max_gradient_difference = max(
        (gradients[name] - parameters[name].grad).abs().max().item()
        for name in parameters
    )
    if (
        not torch.isfinite(torch.tensor(max_gradient_difference))
        or max_gradient_difference > 1e-5
    ):
        raise AssertionError(
            f"Manual BPTT disagrees with autograd: {max_gradient_difference:.2e}"
        )

    total_grad_norm = torch.sqrt(
        sum(gradient.square().sum() for gradient in gradients.values())
    )
    clip_scale = min(1.0, max_grad_norm / (total_grad_norm.item() + 1e-6))
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter -= learning_rate * clip_scale * gradients[name]
            gradients[name].zero_()
            parameter.grad = None

    updated_logits, _, _ = forward(inputs, parameters)
    updated_loss = torch.nn.functional.cross_entropy(
        updated_logits.reshape(-1, output_size), targets.reshape(-1)
    )
    if not torch.isfinite(updated_loss):
        raise AssertionError("Loss became non-finite after the parameter update")

    print(f"Loss before update:       {loss.item():.6f}")
    print(f"Loss after update:        {updated_loss.item():.6f}")
    print(f"Total gradient norm:      {total_grad_norm.item():.6f}")
    print(f"Gradient clipping scale:  {clip_scale:.6f}")
    print(f"Max BPTT/autograd delta:  {max_gradient_difference:.2e}")
    print("Gradient accumulators reset: yes")


if __name__ == "__main__":
    main()
