
import torch
import torch.nn as nn
import time

torch.manual_seed(42)
device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))

# Generate training data
x = torch.randn(8192, 2, device=device)
y = 3 * x[:, :1] + 2 * x[:, 1:] + 1

# Neural network
model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.003
)

loss_fn = nn.MSELoss()

torch.cuda.synchronize()
start = time.time()

# Train AI
for epoch in range(501):
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

torch.cuda.synchronize()

print("\nTraining completed!")
print("Time:", round(time.time() - start, 2), "seconds")
print("Final loss:", loss.item())
print("Peak VRAM:", round(
    torch.cuda.max_memory_allocated() / 1024**2, 2
), "MB")

with torch.no_grad():
    test = torch.tensor([[2.0, 3.0]], device=device)
    print("AI prediction:", model(test).item())
    print("Expected: 13.0")
