import torch
ei_dist = torch.tensor([
        -21.2421, -23.4450, -27.5351, -25.4618, -29.8107, -14.9128, -54.1605,
        -74.9977, -21.0426, -82.6769,  -6.7310, -83.1386,  -4.7584, -84.5476,
         -2.8160, -85.5550,   0.0000, -88.7420,  -2.9497, -85.7527,  -1.7362,
        -87.2520, -21.5643, -87.9634, -21.6701, -87.4041, -22.3665, -87.7528,
        -21.9607, -87.3858, -21.6816])
z_ui = torch.sum(torch.exp(-1*ei_dist), dim=-1) #Size(1) #inf很有可能都是因为ei_dist里有element超过了-88
print(f'z_ui before: {z_ui}')
if torch.isinf(z_ui):
        ei_dist = torch.where(torch.isinf(ei_dist), torch.tensor(-85), ei_dist)
        z_ui = torch.sum(torch.exp(-1*ei_dist), dim=-1)
        print(f'z_ui after: {z_ui}')
expanded_z_ui = z_ui.unsqueeze(0).expand_as(ei_dist) #Size(num_classes)?
logp_xui = torch.log(torch.exp(-1*ei_dist) / expanded_z_ui)
print(f'torch.exp(-1*ei_dist): {torch.exp(-1*ei_dist)}')
print(f'z_ui: {z_ui}')
assert torch.isnan(logp_xui).any() == False