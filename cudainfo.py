import torch
print ('Avaiable: ' + str(torch.cuda.is_available()))
print ('Quantidade: ' + str(torch.cuda.device_count()))
print ('Nome: ' + str(torch.cuda.get_device_name(0)))