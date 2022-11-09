import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('IMGS/img1.jpg', 0)
template = cv.imread('IMGS/template.jpg', 0)

titles = ['Исходное', 'Шаблон']
images = [img, template]

#for i in range(2):
#    plt.subplot(1,2,i+1)
#    plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
#    plt.title(titles[i])
#    plt.xticks([])
#    plt.yticks([])
#plt.show()

map = cv.matchTemplate(img, template, cv.TM_CCOEFF)# карта вероятность меньше чем исходное изображение

#plt.imshow(map, cmap='gray')
#plt.xticks([])
#plt.yticks([])
#plt.title('Карта вероятности')
#plt.show()

_, _, _, max_loc = cv.minMaxLoc(map)

top_left = max_loc

bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

cv.rectangle(img, top_left, bottom_right, 255, 10)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title('Результат')
plt.show()