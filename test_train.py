import json
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 加载并处理数据
with open('test.json', 'r') as f:
    data_test = json.load(f)

test_data = np.array(data_test['Test Data'], dtype=np.float32)
test_data = torch.from_numpy(test_data)


# 创建模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.relu(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


# 定义参数
input_size = 2
hidden_size = 4
output_size = 2

# 实例化RNN模型
rnn = RNN(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    for i in range(test_data.size(0)):
        output, hidden = rnn(test_data[i].unsqueeze(0), hidden)
    loss = criterion(output.squeeze(0), test_data[-1])
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print('Epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss.item()))

# 测试模型
hidden = rnn.initHidden()
for i in range(test_data.size(0)):
    output, hidden = rnn(test_data[i].unsqueeze(0), hidden)

print('Predicted value:', output.detach().numpy())

# 绘制预测结果
plt.plot(test_data.numpy(), 'ro-', label='Real')
plt.plot(output.detach().numpy(), 'bo-', label='Predicted')
plt.legend()
plt.show()
