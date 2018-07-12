import cv2
import numpy as np


def lerdados():
    img = cv2.imread('aero_image.jpeg')
    return img



def lerdGrey():
    img = cv2.imread('red_circles.jpeg',0)
    return img

#def filtro_rgb():


def mascara_vermelho(img):
    blur = cv2.medianBlur(img,5)
    new_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(new_img,np.array([0,100,100]),np.array([10,255,255]))
    mask2 = cv2.inRange(new_img,np.array([160,100,100]),np.array([179,255,255]))
    dst = cv2.addWeighted(mask1,1,mask2,1,0)
    mostrar_img(dst)
    return dst

def mascara_branca(img):
    blur = cv2.medianBlur(img,5)
    new_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,220], dtype=np.uint8)
    upper_white = np.array([255,30,255], dtype=np.uint8)
    dst = cv2.inRange(new_img,lower_white,upper_white)
    mostrar_img(dst)
    return dst


def mostrar_img(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    k= cv2.waitKey(0)


#def detecta_formas_simple():

def detecta_circ(img,cimg):
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,1, param1=20,param2=10,minRadius=5,maxRadius=15)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2]+25,(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    return cimg

def main():
    img = lerdados()
    save = img
    img = mascara_branca(img)
    cimg = detecta_circ(img,save)
    mostrar_img(cimg)
    cv2.imwrite("x.png",cimg)

main()
