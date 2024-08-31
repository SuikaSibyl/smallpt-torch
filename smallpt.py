import os; os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import imageio; import matplotlib.pyplot as plt; import tqdm
import torch; chosen_device = 'cuda'

# ray-sphere intersection with torch tensors
def ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius):
  op = sphere_center - ray_origin; eps = 1e-4
  b = torch.matmul(op.unsqueeze(-2), ray_direction.unsqueeze(-1)).squeeze(-1).squeeze(-1)
  c = torch.matmul(op.unsqueeze(-2), op.unsqueeze(-1)).squeeze(-1).squeeze(-1)
  det = b * b - c + sphere_radius**2; neg = det < 0; det[neg] = 0; det = torch.sqrt(det)
  dist = torch.where(b-det>eps, b-det, torch.where(b+det>eps, b+det, 0)); dist[neg] = 0
  return dist

# define the scene, by a list of spheres
nc=1; nt=1.5; a=nt-nc; b=nt+nc; R0=a*a/(b*b)
spheres_radius = torch.tensor([1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 16.5, 16.5, 600], device=chosen_device)
spheres_position = torch.tensor([[1e5+1, 40.8, 81.6], [-1e5+99, 40.8, 81.6], [50, 40.8, 1e5], [50, 40.8, -1e5+170], [50, 1e5, 81.6], [50, -1e5+81.6, 81.6], [27, 16.5, 47], [73, 16.5, 78], [50, 681.6-0.27, 81.6]], device=chosen_device)
spheres_emission = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [12, 12, 12]], device=chosen_device)
sphere_color = torch.tensor([[0.75, 0.25, 0.25], [0.25, 0.25, 0.75], [0.75, 0.75, 0.75], [0, 0, 0], [0.75, 0.75, 0.75], [0.75, 0.75, 0.75], [.999, .999, .999], [.999, .999, .999], [0, 0, 0]], device=chosen_device)
sphere_reflection = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2, 0], device=chosen_device)

# intersect a ray with the scene
def intersect(ray_origin, ray_direction):
  id = torch.ones([ray_origin.shape[0], ray_origin.shape[1]], dtype=torch.int32, device=chosen_device) * -1
  t = torch.ones([ray_origin.shape[0], ray_origin.shape[1]], device=chosen_device) * 1e20
  for i in range(len(spheres_radius)):
    d = ray_sphere_intersect(ray_origin, ray_direction, spheres_position[i], spheres_radius[i])
    id = torch.where((d > 0) & (d < t), i, id); t = torch.where((d > 0) & (d < t), d, t)
  return id, t

# compute the radiance along a path
def radiance(ray_origin, ray_direction):
  throughput = torch.ones_like(ray_origin)
  radiance = torch.zeros_like(ray_origin)
  for _ in range(10): # recursion depth
    id, t = intersect(ray_origin, ray_direction)
    throughput[id == -1] = 0
    x = ray_origin + ray_direction * t.unsqueeze(-1)
    n = (x - spheres_position[id]); n = n / torch.norm(n, dim=-1, keepdim=True)
    nl = torch.where(torch.matmul(n.unsqueeze(-2), ray_direction.unsqueeze(-1)).squeeze(-1) < 0, n, -n)
    f = sphere_color[id]
    # diffuse surfaces
    theta = 2.0 * torch.pi * torch.rand(ray_origin.shape[0:2], device=chosen_device)
    u = 2.0 * torch.rand(ray_origin.shape[0:2], device=chosen_device) - 1.0; r = torch.sqrt(1.0 - u*u); 
    dir = torch.stack([r*torch.cos(theta), r*torch.sin(theta), u], dim=-1); dir = dir + nl
    # specular surfaces
    refl_dir = ray_direction - n * 2.0 * torch.matmul(n.unsqueeze(-2), ray_direction.unsqueeze(-1)).squeeze(-1)
    dir = torch.where(sphere_reflection[id].unsqueeze(-1) == 1, refl_dir, dir)
    # refraction surfaces
    into = torch.matmul(n.unsqueeze(-2), nl.unsqueeze(-1)).squeeze(-1) > 0
    nnt = torch.where(into, nc/nt, nt/nc); ddn = torch.matmul(ray_direction.unsqueeze(-2), nl.unsqueeze(-1)).squeeze(-1)
    cos2t = 1 - nnt * nnt * (1 - ddn * ddn); tir = (cos2t < 0) & (sphere_reflection[id].unsqueeze(-1) == 2) # total internal reflection
    dir = torch.where(tir, refl_dir, dir)
    tmp = ddn*nnt+torch.sqrt(cos2t); tmp = torch.where(into, tmp, -tmp)
    refr_dir = ray_direction * nnt - n * tmp; refr_dir = refr_dir / torch.norm(refr_dir, dim=-1, keepdim=True)
    c = 1 - torch.where(into, -ddn, torch.matmul(refr_dir.unsqueeze(-2), n.unsqueeze(-1)).squeeze(-1))
    Re=R0+(1-R0)*c*c*c*c*c; Tr=1-Re; P=.25+.5*Re; RP=Re/P; TP=Tr/(1-P) 
    rand = torch.rand(ray_origin.shape[0:2], device=chosen_device).unsqueeze(-1)
    refl = (rand < P) & (sphere_reflection[id].unsqueeze(-1) == 2) & (~tir)
    refr = (rand >= P) &  (sphere_reflection[id].unsqueeze(-1) == 2) & (~tir)
    dir = torch.where(refl, refl_dir, dir); throughput = torch.where(refl, throughput * RP, throughput)
    dir = torch.where(refr, refr_dir, dir); throughput = torch.where(refr, throughput * TP, throughput)

    radiance = radiance + throughput * spheres_emission[id]
    throughput = throughput * f

    n_dired = torch.matmul(n.unsqueeze(-2), dir.unsqueeze(-1)).squeeze(-1)
    n = torch.where(n_dired < 0, -n, n)
    ray_origin = x + n  * 0.025
    ray_direction = dir
    
  return radiance

w=1024; h=768; samps=50; fov = 0.5135
eye = torch.tensor([50, 52, 295.6])
gaze = torch.tensor([0, -0.042612, -1]) / torch.norm(torch.tensor([0, -0.042612, -1]))
cx = torch.tensor([w * fov / h, 0.0, 0.0])
cy = torch.linalg.cross(cx, gaze); cy = cy / torch.norm(cy) * fov
x = torch.arange(w).repeat(h, 1).float()
y = torch.arange(h).repeat(w, 1).t().float().flip(0)
u1 = 2.0 * torch.rand_like(x); dx = torch.where(u1 < 1, torch.sqrt(u1) - 1.0, 1.0 - torch.sqrt(2.0 - u1))
u2 = 2.0 * torch.rand_like(x); dy = torch.where(u2 < 1, torch.sqrt(u2) - 1.0, 1.0 - torch.sqrt(2.0 - u2))
d = (((0.5 + dx) + x) / w - 0.5).unsqueeze(-1) * cx.view(1,1,3) + \
    (((0.5 + dy) + y) / h - 0.5).unsqueeze(-1) * cy.view(1,1,3) + gaze.view(1,1,3)
d = d / torch.norm(d, dim=-1, keepdim=True)
origin = eye + d * 140; d = d / torch.norm(d, dim=-1, keepdim=True)

origin = origin.to('cuda'); d = d.to('cuda')
L = torch.zeros([h, w, 3], device=chosen_device)
with torch.no_grad():
  for i in tqdm.tqdm(range(samps)):
    L = L + radiance(origin, d)
L = L / samps

plt.imshow(L.cpu().numpy())
plt.show()

# Write to disk
imageio.imwrite('cornell-box.exr', L.cpu().numpy())