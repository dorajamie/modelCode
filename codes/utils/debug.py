import math

import torch



a = torch.tensor([1,0.0976,0.765,0,0.0000001],dtype=torch.float)

b =torch.abs(torch.tan(torch.mul(torch.add(a,-1),math.pi / 2)))

print(b)