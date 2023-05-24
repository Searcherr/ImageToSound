import numpy as np
import thinkdsp
import cv2
import thinkplot
from PIL import Image


class ImageToSoundConverter:
    def __init__(self, image_path):

        self.image_path = image_path
        self.image = None
        self.pixels = None
        self.wave = None
        self.spectrum = None

    def read_image(self):
        self.image = cv2.imread(self.image_path)
        # Convert to grayscale
        if self.image is not None:
            self.pixels = np.mean(self.image, axis=2)

    def grayscale_image(self):
        if self.image is not None:
            grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.pixels = np.array(grayscale_image)

    def show_image(self):
        if self.image is not None:
            cv2.imshow('Your image', self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_grayscaled_image(self):
        if self.pixels is not None:
            cv2.imshow('Your image', self.pixels)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def to_wave(self, duration=1.0, framerate=44100):
        if self.pixels is not None:
            num_samples = int(duration * framerate)
            ts = np.linspace(0, duration, num_samples, endpoint=False)
            ys = self.pixels.flatten() / 255.0  # Normalize pixel values to range [0, 1]

            self.wave = thinkdsp.Wave(ys=ys, ts=ts, framerate=framerate)

    def create_spectrum(self):
        if self.wave is not None:
            self.spectrum = self.wave.make_spectrum()

    def plot_spectrum(self):
        if self.spectrum is not None:
            self.spectrum.plot()
            thinkplot.show()

    def create_sound(self, output_path):
        if self.wave is not None:
            self.wave.write(filename=output_path)


if __name__ == "__main__":
    image_to_sound = ImageToSoundConverter("./Images/samoyed-few-colors.jpg")
    image_to_sound.read_image()
    image_to_sound.grayscale_image()
    #print(image_to_sound.image)
    image_to_sound.show_image()
    image_to_sound.show_grayscaled_image()
    image_to_sound.to_wave()
    image_to_sound.create_spectrum()
    image_to_sound.plot_spectrum()
    image_to_sound.create_sound("./Sounds/sound.wave")
    print(type(image_to_sound.wave))



