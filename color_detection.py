import os

from colorthief import ColorThief


for filename in os.listdir('res/Movie_Poster_Dataset/2015'):
    if filename.endswith(".jpg"):
        print('res/Movie_Poster_Dataset/2015' +filename)
        img = ColorThief('res/Movie_Poster_Dataset/2015/'+filename)
        dominant_color = img.get_color(quality=1)
        print(dominant_color)
        palette = img.get_palette(color_count=6)
        print(palette)

        continue
    else:
        continue