from Task1.template_matching import TemplateMatching
from Task1.imageFunctions import display
from PIL import Image


source = Image.open('rotated_low_cloaked.png')
template = Image.open('template.png')
tm = TemplateMatching(source)

found = tm.find_one(template, 20, 100)
if found is not None:
    drawn = display(source, [found])
    drawn.save('res_fawkes.png')

