import cv2 #library for the number plate detection
import matplotlib.pyplot as plt #library for the marking around the number plate
#cascade that using the pretrained code for the number plate detection
#This cascade classifier for the ML for number plate detection 
#giving the path to the opencv
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml') 

def detect_number_plate(image):#function getting the image
    img = cv2.imread(image)#it will read the image file
    #it will convert the image to grayscale to increase the speed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #it is used to detect the object in the image
    plates = cascade.detectMultiScale(gray, 1.1, 10)

    #for draw rectangle x,y top left corner
    #w,h is width and height
    for (x, y, w, h) in plates:
        #draw rectangel in the image //bottom right corner // color // thickness
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#convert to the RGB
    plt.imshow(img_rgb)#show image with plot
    plt.axis('off')
    plt.show()#show image

image = 'download3.jpg' # giving the path of the image 
detect_number_plate(image) #passing the image to the function
