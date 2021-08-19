import imageio
image_list = [r'./CGAN/log/' + 'sample_' + str(x) + ".png" for x in range(50)]
gif_name = r'cgan.gif'

frames = []
for image_name in image_list:
    frames.append(imageio.imread(image_name))
imageio.mimsave(gif_name, frames, 'GIF', duration=0.3)