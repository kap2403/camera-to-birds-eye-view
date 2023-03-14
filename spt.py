import torch
import torch.nn as nn
from utils import device
class BilinearInterpolation(nn.Module):
    
    def __init__(self):
        super(BilinearInterpolation, self).__init__()
    
    def interpolate(self,image, sampled_grids):
        batch_size = image.shape[0]
        height = image.shape[2]
        width = image.shape[3]
        num_channels = image.shape[1]

        x = torch.flatten(sampled_grids[:, 0:1, :])
        #print('x',x)
        x = x.to(torch.float64)
        y = torch.flatten(sampled_grids[:, 1:2, :])
        #print('y',x)
        y = y.to(torch.float64)

    
        x = (.5 * (x + 1.0) * width)
        #print('x+width',x)
        x = x.to(torch.float32)
        y = (.5 * (y + 1.0) * height)
        #print('y+width',y)
        y = y.to(torch.float32)
        

        x0 = (x.round()).type(torch.int32)
        #print('x0',x0)
        x1 = x0 + 1
        y0 = (y.round()).type(torch.int32)
        #print('y0',y0)
        y1 = y0 + 1

        max_x = int(image.shape[3] - 1)
        #print('max_x',max_x)
        max_y = int(image.shape[2] - 1)
        #print('max_y',max_y)
        

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        pixels_batch = torch.arange(0, batch_size) * (height * width)
        pixels_batch = pixels_batch.unsqueeze(-1)
        #print('pixels_batch',pixels_batch)
        flat_output_size = height*width
        base = pixels_batch.repeat(1, flat_output_size)
        #print('base',base)
        base = base.flatten()

        base_y0 = y0 * width
        base_y0 = base + base_y0
        base_y1 = y1 * width
        base_y1 = base_y1 + base
        #print('baea',base_y0+x0)
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1
        #print('indices',indices_a.max(),indices_b.max(),indices_c.max(),indices_d.max())
        flat_image = image.reshape(-1, num_channels)
        flat_image = flat_image.to(torch.uint8)

        pixel_values_a = flat_image[indices_a].to(device=device) 
        pixel_values_b = flat_image[indices_b].to(device=device) 
        pixel_values_c = flat_image[indices_c].to(device=device) 
        pixel_values_d = flat_image[indices_d].to(device=device) 
        
        
         #   print("Invalid index detected")
            
        x0 = x0.to(torch.float32)
        x1 = x1.to(torch.float32)
        y0 = y0.to(torch.float32)
        y1 = y1.to(torch.float32)

        area_a = (((x1 - x) * (y1 - y)).unsqueeze(1)).to(device=device) 
        area_b = (((x1 - x) * (y - y0)).unsqueeze(1)).to(device=device) 
        area_c = (((x - x0) * (y1 - y)).unsqueeze(1)).to(device=device) 
        area_d = (((x - x0) * (y - y0)).unsqueeze(1)).to(device=device) 

        values_a = (area_a * pixel_values_a).to(device=device) 
        values_b = (area_b * pixel_values_b).to(device=device) 
        values_c = (area_c * pixel_values_c).to(device=device) 
        values_d = (area_d * pixel_values_d).to(device=device) 
        return values_a + values_b + values_c + values_d
    
    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = torch.linspace(-1., 1., width)
        y_linspace = torch.linspace(-1., 1., height)
        x_coordinates, y_coordinates = torch.meshgrid(x_linspace, y_linspace)
        x_coordinates = x_coordinates.flatten()
        y_coordinates = y_coordinates.flatten()
        ones = torch.ones_like(x_coordinates)
        grid = torch.cat([x_coordinates, y_coordinates, ones], 0)

        grid = grid.flatten()
        grids = torch.cat([grid] * batch_size, dim=0)
        return grids.reshape(batch_size, 3, height * width)
    
    def forward(self, x,affine_transformation):
        
        batch_size, num_channels,height, width = x.size()

        affine_transformation = affine_transformation.repeat(batch_size,1,1)

        transformation = affine_transformation.view(batch_size, 3, 3)
        
        regular_grids = self._make_regular_grids(batch_size, height, width)
 
        sampled_grids = torch.bmm(regular_grids.permute(0,2,1), transformation)
    
        sampled_grids = sampled_grids.permute(0,2,1)
       
        w = 1 / sampled_grids[:, 2, :]
        w = w.reshape(batch_size, 1, w.shape[-1])
        w = torch.tile(w, [1, 2, 1])
        w = w.expand(batch_size, 2, w.shape[-1])
        sampled_grids = sampled_grids[:, 0:2, :] * w
        
        #print('samp_grid',sampled_grids)
        
        interpolated_image = self.interpolate(x,sampled_grids)
        #print('inter',interpolated_image)
        interpolated_image = interpolated_image.view(batch_size, num_channels,height, width)
        #print(interpolated_image)
                                                     
        
        return interpolated_image#.to(torch.uint8)



