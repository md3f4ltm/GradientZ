#!/usr/bin/env python3
"""
PyTorch comparison script for GradientZ autograd examples.
This script runs the same autograd tests as in GradientZ to compare results.
"""

import torch
import torch.nn.functional as F
import numpy as np

def print_tensor(name, tensor):
    """Helper function to print tensors in a readable format"""
    print(f"{name}:")
    if tensor.dim() == 0:
        print(f"{tensor.item():.4f}")
    elif tensor.dim() == 1:
        print(f"[{', '.join(f'{x:.4f}' for x in tensor)}]")
    elif tensor.dim() == 2:
        print("[")
        for row in tensor:
            print(f"  [{', '.join(f'{x:.4f}' for x in row)}]")
        print("]")
    else:
        print(tensor)
    print()

def test1_simple_operations():
    """Test 1: Simple operations (a + b) * c"""
    print("=" * 50)
    print("Test 1: Simple operations (a + b) * c")

    # Create tensors with gradients enabled
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True)

    print_tensor("a", a)
    print_tensor("b", b)
    print_tensor("c", c)

    # Forward pass: (a + b) * c
    result = (a + b) * c
    print_tensor("result = (a + b) * c", result)

    # Backward pass
    result.backward(torch.ones_like(result))

    print("Gradients after backward():")
    print_tensor("grad_a", a.grad)
    print_tensor("grad_b", b.grad)
    print_tensor("grad_c", c.grad)

def test2_power_operation():
    """Test 2: Power operation x^2 - y"""
    print("=" * 50)
    print("Test 2: Power operation x^2 - y")

    x = torch.tensor([[2.0, 3.0, -1.0]], requires_grad=True)
    y = torch.tensor([[1.0, 4.0, 2.0]], requires_grad=True)

    print_tensor("x", x)
    print_tensor("y", y)

    # Forward pass: x^2 - y
    result = x.pow(2) - y
    print_tensor("result = x^2 - y", result)

    # Backward pass
    result.backward(torch.ones_like(result))

    print("Gradients:")
    print_tensor("grad_x (should be 2*x = [4, 6, -2])", x.grad)
    print_tensor("grad_y (should be [-1, -1, -1])", y.grad)

def test3_relu_activation():
    """Test 3: ReLU activation"""
    print("=" * 50)
    print("Test 3: ReLU activation")

    input_tensor = torch.tensor([[-2.0, -1.0, 1.0, 2.0]], requires_grad=True)

    print_tensor("input", input_tensor)

    # Forward pass: ReLU
    output = F.relu(input_tensor)
    print_tensor("ReLU(input)", output)

    # Backward pass
    output.backward(torch.ones_like(output))

    print("Gradients:")
    print_tensor("grad_input (should be [0, 0, 1, 1])", input_tensor.grad)

def test4_tanh_activation():
    """Test 4: Tanh activation"""
    print("=" * 50)
    print("Test 4: Tanh activation")

    input_tensor = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)

    print_tensor("input", input_tensor)

    # Forward pass: tanh
    output = torch.tanh(input_tensor)
    print_tensor("tanh(input)", output)

    # Backward pass
    output.backward(torch.ones_like(output))

    print("Gradients:")
    print_tensor("grad_input (tanh derivative: 1 - tanh^2)", input_tensor.grad)

def test5_complex_chain_rule():
    """Test 5: Complex chain rule: sigmoid(x * w + b)"""
    print("=" * 50)
    print("Test 5: Complex chain rule: sigmoid(x * w + b)")

    weight = torch.tensor([[0.5, -0.3], [0.2, 0.7]], requires_grad=True)
    bias = torch.tensor([[0.1, -0.1], [0.0, 0.2]], requires_grad=True)
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], requires_grad=True)

    print_tensor("weight", weight)
    print_tensor("bias", bias)
    print_tensor("x", x)

    # Forward pass: sigmoid(x * w + b)
    output = torch.sigmoid(x * weight + bias)
    print_tensor("output = sigmoid(x * w + b)", output)

    # Backward pass
    output.backward(torch.ones_like(output))

    print("Gradients:")
    print_tensor("grad_weight", weight.grad)
    print_tensor("grad_bias", bias.grad)
    print_tensor("grad_x", x.grad)

def test6_gradient_accumulation():
    """Test 6: Gradient accumulation (y = x + x)"""
    print("=" * 50)
    print("Test 6: Gradient accumulation (y = x + x)")

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    print_tensor("x", x)

    # Forward pass: y = x + x
    y = x + x
    print_tensor("y = x + x", y)

    # Backward pass
    y.backward(torch.ones_like(y))

    print("Gradients:")
    print_tensor("grad_x (should be [2, 2, 2, 2] due to accumulation)", x.grad)

def test7_simple_neural_network():
    """Test 7: Simple neural network forward and backward"""
    print("=" * 50)
    print("Test 7: Simple neural network forward and backward")

    # Input: [1.0, -0.5]
    input_data = torch.tensor([1.0, -0.5], requires_grad=True)

    # Weight matrix (2x2)
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)

    # Bias vector
    bias = torch.tensor([0.1, -0.1], requires_grad=True)

    print(f"Input: {input_data.tolist()}")
    print_tensor("Weight matrix", weight)
    print_tensor("Bias vector", bias)

    # Forward pass: linear transformation followed by sigmoid
    linear_output = torch.matmul(input_data, weight) + bias
    output = torch.sigmoid(linear_output)

    print_tensor("Linear output (input @ weight + bias)", linear_output)
    print_tensor("Final output (sigmoid)", output)

    # Create a simple loss (sum of outputs)
    loss = output.sum()
    print(f"Loss (sum of outputs): {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print("\nGradients:")
    print_tensor("grad_input", input_data.grad)
    print_tensor("grad_weight", weight.grad)
    print_tensor("grad_bias", bias.grad)

def main():
    """Run all PyTorch autograd tests"""
    print("🔥 PyTorch Autograd Comparison Tests")
    print("These tests mirror the GradientZ autograd examples")
    print("=" * 50)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run all tests
    test1_simple_operations()
    test2_power_operation()
    test3_relu_activation()
    test4_tanh_activation()
    test5_complex_chain_rule()
    test6_gradient_accumulation()
    test7_simple_neural_network()

    print("\n" + "=" * 50)
    print("🎯 PyTorch Comparison Tests Completed!")
    print("\nKey observations for comparison with GradientZ:")
    print("1. Gradient values should match (within floating-point precision)")
    print("2. Chain rule application should be identical")
    print("3. Gradient accumulation behavior should be the same")
    print("4. Activation function derivatives should match")
    print("5. Matrix operations gradients should be equivalent")

if __name__ == "__main__":
    main()
